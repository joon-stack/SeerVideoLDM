from functools import partial, wraps
from .data_utils import cast_num_frames, list_to_tensor, identity
import torch.nn.functional as F
from torchvision import transforms as T
import os.path as osp
import json
from torch.utils.data import Dataset
from torch.utils import data
from pathlib import Path
import numpy as np
import glob
import os

from PIL import Image, ImageFile
import torch


def glob_all(dir_path, only_dir=False, sort=True, strip=True):
    """Similar to `scandir`, but return the entire path cat with dir_path."""
    if only_dir:
        pattern = osp.join(dir_path, '*/')
    else:
        pattern = osp.join(dir_path, '*')
    results = glob.glob(pattern)
    if sort:
        results.sort()
    if strip:
        results = [res.rstrip('/') for res in results]
    return results


class Dataset(data.Dataset):
    def __init__(
        self,
        data_root,
        image_size,
        split = 'train',
        n_sample_frames=16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['png'],
        normalize = True,
    ):
        super().__init__()

        assert split in ['train', 'val', 'test']

        self.data_root = osp.join(data_root, split)
        self.split = split

        self.image_size = image_size
        self.video_len = 50
        self.frame_offset = 1
        self.n_sample_frames = n_sample_frames

        self.valid_idx = self._get_sample_idx()

        self.inst_root = os.path.join(data_root, "labels")
        inst = np.load(os.path.join(self.inst_root, "inst.npy"))
        self.inst = torch.from_numpy(inst).float()
    
        
        if normalize:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
                # T.CenterCrop(self.image_size,),
                
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
                # T.CenterCrop(self.image_size),
                
            ])

    def _get_sample_idx(self):
        valid_idx = []  # (video_folder, start_idx)
        files = glob_all(self.data_root, only_dir=True)
        self.files = [s.rstrip('/') for s in files]
        valid_files = []
        self.num_videos = len(self.files)
        for folder in self.files:
            # simply use random uniform sampling
            if self.split == 'train':
                # for trajectories shorter than video_len
                frame_len = len(os.listdir(folder))
                max_start_idx = min(frame_len, self.video_len) - \
                    (self.n_sample_frames - 1) * self.frame_offset
                #valid_idx += [(folder, idx) for idx in range(max_start_idx)]

                if frame_len > self.video_len:
                    valid_files.append(folder)
                    valid_idx += [(folder, idx) for idx in range(max_start_idx)]
                

            # only test once per video
            else:
                frame_len = len(os.listdir(folder))

                if frame_len > self.video_len:
                    valid_idx += [(folder, 0)]
                    valid_files.append(folder)
                    
        self.files = valid_files
        self.num_videos = len(self.files)
        return valid_idx
    
    def _get_video_start_idx(self, idx):
        return self.valid_idx[idx]

    def _decode_inst(self, inst):
        """Utlity to decode encoded language instruction"""
        return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")

    def _read_frames(self, idx):
        folder, start_idx = self._get_video_start_idx(idx)
        start_idx += 1  # files start from 'test_1.png'
        filename = osp.join(folder, 'test_{}.png')
        
        frames = [
            Image.open(filename.format(start_idx +
                                       n * self.frame_offset)).convert('RGB')
            for n in range(self.n_sample_frames)
        ]

        frames = [self.transform(img) for img in frames]
        return torch.stack(frames, dim=0).transpose(0, 1)  # [C, N, H, W]
    
    def _read_insts(self, idx):
        folder, _ = self.valid_idx[idx]
        return self.inst[int(os.path.basename(folder))]
    
    def _read_insts_text(self, idx):
        folder, _ = self.valid_idx[idx]
        inst_file = os.path.join(self.inst_root, os.path.basename(folder)+".npy")
        return self._decode_inst(np.load(inst_file))

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, index):
        frames = self._read_frames(index)
        insts = self._read_insts_text(index)
        return frames, insts