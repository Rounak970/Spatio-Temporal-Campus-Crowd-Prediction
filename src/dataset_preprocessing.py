"""
dataset_preprocessing.py
------------------------------------
Utilities for preprocessing video frames
and preparing them as input for the 3D CNN + ConvLSTM model.
"""

import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------
# 1. Load & preprocess all frames
# --------------------------------------------------------
def load_frames(folder_path):
    frames = []
    for name in sorted(os.listdir(folder_path)):
        if not name.lower().endswith((".jpg", ".png")):
            continue

        img = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = img.astype("float32") / 255.0
        frames.append(img)

    return np.array(frames)


# --------------------------------------------------------
# 2. Make sequential 10-frame clips
# --------------------------------------------------------
def make_clips(frames, clip_len=10):
    clips = []
    for i in range(len(frames) - clip_len + 1):
        clip = frames[i:i + clip_len]
        clips.append(clip)
    return np.array(clips)


# --------------------------------------------------------
# 3. PyTorch Dataset Wrapper
# --------------------------------------------------------
class CrowdDataset(Dataset):
    def __init__(self, clips, wifi, counts, densities, motions):
        self.clips = clips
        self.wifi = wifi
        self.counts = counts
        self.densities = densities
        self.motions = motions

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video = torch.tensor(self.clips[idx]).unsqueeze(0)  # → (1, 10, 64, 64)
        wifi = torch.tensor(self.wifi[idx])
        count = torch.tensor(self.counts[idx], dtype=torch.float32)
        density = torch.tensor(self.densities[idx], dtype=torch.float32)
        motion = torch.tensor(self.motions[idx], dtype=torch.float32)
        return video, wifi, count, density, motion
