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
from model import  DCDiscriminator, DCSAGenerator, SwinGenerator


import PIL.Image as Image


class Self_Attn_Base(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn_Base,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out,attention

class Self_Attn_Generator(nn.Module):
    """ Self attention Layer"""
    def __init__(self,attention_dim,activation):
        super(Self_Attn_Generator,self).__init__()
        self.chanel_in = attention_dim
        self.activation = activation

        self.attn_base = Self_Attn_Base(attention_dim, activation)

        self.sigma = 0

    def update_sigma(self, sigma):
        if sigma > 1:
            sigma = 1
        self.sigma = sigma

    def forward(self,x):
      
        out, attention = self.attn_base(x)
        
        out = self.sigma * out + (1 - self.sigma) * x 
        return out, attention

class DCGenerator(nn.Module):
    def __init__(self, z_dim = 512, ngf = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( z_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, 16, 4, 2, 1, bias=False),
            # state size. 3 x 64 x 64
        )

        self.attn = Self_Attn_Generator(16, 'relu')
        self.toRGB = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False)
        # 3 x 128 x 128
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.main(z)
        #x, attn = self.attn(x)
        x = self.toRGB(x)
        x = self.tanh(x)
        return x#, attn

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

def getSWINGAN():
    netG = SwinGenerator(z_dim = latent_dim, embed_dim=96).to(device)
    model_save_path = './models/swingan - 3'
    pretrained_model = '180000'

    netG.load_state_dict(torch.load(os.path.join(
            model_save_path, '{}_G.pth'.format(pretrained_model))))
    return netG

def getModel(model_name):
    if model_name == 'DCGAN':
        return getDCGAN()
    elif model_name == 'DCSAGAN':
        return getDCSAGAN()
    elif model_name == 'PGAN':
        return getPGAN()
    elif model_name == 'DC2SAGAN':
        return getDC2SAGAN()
    elif model_name == "SWINGAN":
        return getSWINGAN()

ignite.utils.manual_seed(36)

print(idist.device())
fid_metric = FID(device=idist.device())

batch_size = 16
latent_dim = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'SWINGAN'
netG = getModel(model_name)


def evaluation_step(engine, batch):
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim, device=idist.device())
        netG.eval()
        if model_name == 'DCGAN' or model_name == 'PGAN' or model_name == 'SWINGAN':
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

num_batches = 40
idxs = np.arange(num_batches * batch_size)
subset = torch.utils.data.Subset(dataset, idxs)

dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)


#evaluator.run(dataloader)
#metrics = evaluator.state.metrics

#print("FID:", metrics['fid'])

# inception score - modified 

#inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

'''for i, batch in enumerate(dataloader):
    batch = interpolate(batch)
    image = batch[0].to(idist.device())'''


noise = torch.randn(1, latent_dim, device=idist.device())
img = netG(noise)[0]


# denorm image
img = (img - img.min()) / (img.max() - img.min()) 
img = img.permute(1,2,0).detach().cpu()
# plot images
plt.imshow(img.numpy())
plt.show()

pass