import argparse
import itertools
import math
import os
import os.path as osp
import random
from pathlib import Path
from typing import Optional
from einops import rearrange, repeat, reduce
import json
from functools import partial, wraps
from multiprocessing import Process

import numpy as np
import torch
from torchvision import transforms as T
from torch.utils import data
from pathlib import Path
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from seer.models.unet_3d_condition import SeerUNet, FSTextTransformer
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from ldm.models.diffusion.ddim_video import DDIMSampler
from utils.ddim_sampling_utils import ddim_sample, save_visualization_onegif
from omegaconf import OmegaConf

logger = get_logger(__name__)

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    global_step = 0
    # Load models and create wrapper for stable diffusion
    text_tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    sunet = SeerUNet.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        low_cpu_mem_usage = False,
    )
    fstext_model = FSTextTransformer(num_frames = 16, num_layers = 8)
    fstext_model.set_numframe(args.num_frames)
    sampler = DDIMSampler(device = accelerator.device)
    if is_xformers_available():
        try:
            sunet.enable_xformers_memory_efficient_attention()
            fstext_model.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    
    sunet, fstext_model = accelerator.prepare(
        sunet, fstext_model
    )
    load_path = os.path.join(args.output_dir, f"learned_sdunet-steps-{args.saved_global_step}")
    load_path_file = os.path.join(args.output_dir, f"learned_sdunet-steps-{args.saved_global_step}.pt")
    if os.path.exists(load_path):
        print("loading")
        fstext_state_dict = torch.load(os.path.join(load_path,'pytorch_model_1.bin'), map_location="cpu")
        msg = fstext_model.load_state_dict(fstext_state_dict, strict=True)
        print(msg)
        sunet_state_dict = torch.load(os.path.join(load_path,'pytorch_model.bin'), map_location="cpu")
        msg = sunet.load_state_dict(sunet_state_dict, strict=True)
        print(msg)
    if os.path.exists(load_path_file):
        print("loading steps")
        state_dict = torch.load(load_path_file)
        global_step = state_dict["global_step"]


    # Freeze all models
    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(sunet.parameters())
    freeze_params(fstext_model.parameters())
    # Move vae and text encoder to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Keep all models in eval model during inference
    vae.eval()
    text_encoder.eval()
    sunet.eval()
    fstext_model.eval()
    
    transform = T.Compose([
                T.Resize((args.resolution,args.resolution)), #T.CenterCrop(args.resolution),
                T.ToTensor()
            ])
    image_path_list = (args.image_path).split('|')
    x0_image_val_list = []
    for image_path in image_path_list:
        x0_image = Image.open(image_path)
        x0_image = transform(x0_image).to(accelerator.device)
        x0_image = 2.*x0_image - 1.
        x0_image = x0_image.unsqueeze(0).unsqueeze(2)
        x0_image_val_list.append(x0_image)
    x0_image_val = torch.cat(x0_image_val_list,dim=2)
    num_samples = x0_image_val.shape[0]
    cond_tokens_val = [args.input_text_prompts] * num_samples

    cond_input = text_tokenizer(
                cond_tokens_val,
                padding="max_length",
                max_length=text_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
    text_cond_emb_val = text_encoder(
            cond_input.input_ids.to(accelerator.device),
            attention_mask=cond_input.attention_mask.to(accelerator.device),
    )
    text_cond_emb_val = text_cond_emb_val[0]
    cond_input_empty = text_tokenizer(
                len(cond_tokens_val)*[''],
                padding="max_length",
                max_length=text_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
    text_empty_emb_val = text_encoder(
            cond_input_empty.input_ids.to(accelerator.device),
            attention_mask=cond_input_empty.attention_mask.to(accelerator.device),
    )
    text_empty_emb_val = text_empty_emb_val[0]
    f1 = x0_image_val.shape[2]
    f2 = args.num_frames-f1
    x0_image_val = x0_image_val.expand(-1,-1,f1,-1,-1)
    x0_image_val = rearrange(x0_image_val, 'b c f h w -> (b f) c h w')
    latents_x0_val = vae.encode(x0_image_val).latent_dist.sample().detach()
    latents_x0_val = latents_x0_val * 0.18215
    latents_x0_val = rearrange(latents_x0_val, '(b f) c h w -> b c f h w', f=f1)
    b,c_l,_,h_l,w_l = latents_x0_val.shape
    x0_image_val = rearrange(x0_image_val, '(b f) c h w -> b c f h w', f=f1)
    x0_image_val = (x0_image_val+ 1.0) / 2.0

    text_seq_cond_emb = fstext_model(context=text_cond_emb_val).detach()
    text_empty_emb_val = text_empty_emb_val.unsqueeze(1).expand(-1,text_seq_cond_emb.shape[1],-1,-1)
    
    f = f1+f2
    noise_val = torch.randn((b,c_l,f2,h_l,w_l)).to(latents_x0_val.device)
    for j in range(args.num_samples):
        x_samples_ddim = ddim_sample(sampler, sunet, vae, shape=(b,c_l,f2,h_l,w_l), c=text_seq_cond_emb, start_code=noise_val, x0_emb=latents_x0_val,\
                                        ddim_steps=args.ddim_steps, scale=args.scale, uc=text_empty_emb_val)

        save_visualization_onegif(accelerator, vae, x_samples_ddim, x0_image=x0_image_val,\
                            sample_id=j, image_path=(args.image_path).split('|')[0], num_sample_rows = 1)

        noise_val = torch.randn((b,c_l,f2,h_l,w_l)).to(latents_x0_val.device)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    parser.add_argument("--image_path", type=str, default="./src/fig/0.jpg")
    parser.add_argument("--input_text_prompts", type=str, default="")
    args0 = parser.parse_args()
    args = OmegaConf.load(args0.config)
    args.input_text_prompts = args0.input_text_prompts
    args.image_path = args0.image_path
    main(args)