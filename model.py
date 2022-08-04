import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import copy
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
from custom_layers import EqualizedLinear


class Self_Attn_Base(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn_Base,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.ones(1))

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



class Self_Attn_Discriminator(nn.Module):
    def __init__(self,in_dim,activation):
        super(Self_Attn_Discriminator,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.softmax  = nn.Softmax(dim=-1) #
        self.gamma = nn.Parameter(torch.ones(1))

        self.sigma = 0


    def update_sigma(self, sigma):
        if sigma > 1:
            sigma = 1
        self.sigma = sigma

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
        out = self.sigma * out + (1 - self.sigma) * x
        return out,attention


class Generator(nn.Module):
    def __init__(self, pretrained_pgan, attention_dim = 3, use_gpu=True):
        super(Generator, self).__init__()
        self.use_gpu = use_gpu

        # freeze generator parameters
        #       
        for param in pretrained_pgan.netG.parameters():
            param.requires_grad = False
        
        self.netG = copy.deepcopy(pretrained_pgan.netG)

    

        #self.pre_attn = nn.Conv2d(in_channels=3, out_channels=attention_dim, kernel_size=1)
        self.attn = Self_Attn_Generator(attention_dim, 'relu')
        self.toRGB = nn.Conv2d(attention_dim, 3, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.netG(x)
        x, attn = self.attn(x)
        x = self.toRGB(x)
        return x, attn

    
class Discriminator(nn.Module):
    def __init__(self, pretrained_pgan, attention_dim = 3, use_gpu=True):
        super(Discriminator, self).__init__()
        self.use_gpu = use_gpu

        # freeze discriminator parameters
        for param in pretrained_pgan.netD.parameters():
            param.requires_grad = False
        
        pretrained_pgan.netD.decisionLayer = nn.Identity() #Self_Attn(pretrained_pgan.netD.scalesDepth[0], 'relu')
                  
        self.netD = copy.deepcopy(pretrained_pgan.netD)
        if use_gpu:
            self.netD.cuda()
        
        self.attn = Self_Attn_Discriminator(attention_dim, 'relu') # self.netD.scalesDepth[0]

        self.decisionLayer = nn.Conv2d(in_channels = attention_dim , out_channels = 1 , kernel_size= 2)
    
    def forward(self, x):
        x = self.netD(x)
         # ovo treba srediti
        return x
    


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
        x, attn = self.attn(x)
        x = self.toRGB(x)
        x = self.tanh(x)
        return x, attn

class DCDiscriminator(nn.Module):
    def __init__(self, z_dim = 512, ndf = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is 3 x 128 x 128
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
            )
    def forward(self, x):
        x = self.main(x)
        return x
