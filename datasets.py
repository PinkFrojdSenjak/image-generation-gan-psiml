import torch
from torchvision import transforms

import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import imageio


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.747, 0.536, 0.419), (0.205, 0.225, 0.199))])):

        self.root_dir = root_dir
        self.transform = transform
        self.img_list = [f for f in os.listdir(
            self.root_dir) if os.path.splitext(f)[1] == ".jpg"]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = imageio.imread(img_path)
        
        if self.transform:
            img = self.transform(img)
       
        return img

