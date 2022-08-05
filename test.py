import torch
import torchvision
import ignite
from ignite.metrics import FID, InceptionScore

import os
import logging
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import FaceDataset

import torchvision.transforms as transforms
import torchvision.utils as vutils

from ignite.engine import Engine, Events
import ignite.distributed as idist
from model import DCGenerator, DCDiscriminator, DCSAGenerator


import PIL.Image as Image


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

def getDCGAN():
    netG = DCGenerator(latent_dim).to(device)
    model_save_path = './models/psagan_2'
    pretrained_model = '86000'

    netG.load_state_dict(torch.load(os.path.join(
            model_save_path, '{}_G.pth'.format(pretrained_model))))
    return netG

def getDCSAGAN():
    netG = DCSAGenerator(latent_dim).to(device)
    model_save_path = './models/dcsagan - 1'
    pretrained_model = '260000'

    netG.load_state_dict(torch.load(os.path.join(
            model_save_path, '{}_G.pth'.format(pretrained_model))))
    return netG

def getDC2SAGAN():
    netG = DCSAGenerator(latent_dim).to(device)
    model_save_path = './models/sa2gan - 1'
    pretrained_model = ''

    netG.load_state_dict(torch.load(os.path.join(
            model_save_path, '{}_G.pth'.format(pretrained_model))))
    return netG

def getPGAN():
    return torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celeba',
                       pretrained=True, useGPU=True).netG


def getModel(model_name):
    if model_name == 'DCGAN':
        return getDCGAN()
    elif model_name == 'DCSAGAN':
        return getDCSAGAN()
    elif model_name == 'PGAN':
        return getPGAN()
    elif model_name == 'DC2SAGAN':
        return getDC2SAGAN()

ignite.utils.manual_seed(999)

print(idist.device())
fid_metric = FID(device=idist.device())

batch_size = 16
latent_dim = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = ''
netG = getModel(model_name)


def evaluation_step(engine, batch):
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim, device=idist.device())
        netG.eval()
        if model_name == 'DCGAN' or model_name == 'PGAN':
            fake_batch = netG(noise)
            fake = interpolate(fake_batch)
            real = interpolate(batch)
        if model_name == 'DCSAGAN' or model_name == 'DC2SAGAN':
            fake_batch, _ = netG(noise)
            fake = interpolate(fake_batch)
            real = interpolate(batch)
      

        return fake, real

dataset = FaceDataset('img_celeba_cropped')


evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")

num_batches = 20
idxs = np.arange(num_batches * batch_size)
subset = torch.utils.data.Subset(dataset, idxs)

dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)


evaluator.run(dataloader)
metrics = evaluator.state.metrics

print("FID:", metrics['fid'])

# inception score - modified 

inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

for i, batch in enumerate(dataloader):
    batch = interpolate(batch)
    image = batch[0].to(idist.device())

    score = inception(image)

    pass




