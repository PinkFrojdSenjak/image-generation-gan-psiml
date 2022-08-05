import torch
import torchvision
import ignite
from ignite.metrics import FID, InceptionScore

import os
import logging
import matplotlib.pyplot as plt

import numpy as np

from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import FaceDataset

import torchvision.transforms as transforms
import torchvision.utils as vutils

from ignite.engine import Engine, Events
import ignite.distributed as idist
from model import DCGenerator, DCDiscriminator

ignite.utils.manual_seed(999)

print(idist.device())
fid_metric = FID(device=idist.device())

import PIL.Image as Image


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

batch_size = 16
latent_dim = 512

netG = DCGenerator(latent_dim)
model_save_path = './models/dcsagan - 1'
pretrained_model = '0'

netG.load_state_dict(torch.load(os.path.join(
            model_save_path, '{}_G.pth'.format(pretrained_model))))
        #self.D.load_state_dict(torch.load(os.path.join(
        #    self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))

def evaluation_step(engine, batch):
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim, device=idist.device())
        netG.eval()
        fake_batch = netG(noise)
        fake = interpolate(fake_batch)
        real = interpolate(batch[0])
        print('odradjeno')
        return fake, real

dataset = FaceDataset('img_celeba_cropped')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")

evaluator.run()
