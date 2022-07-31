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


class Self_Attn_No_Residual(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn_No_Residual,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

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
        
        return out,attention


class Self_Attn_Generator(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn_Generator,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

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
        x = x.permute(0,2,1,3)
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        x = x.permute(0,2,1,3)
        out = out + x
        return out,attention



class Self_Attn_Discriminator(nn.Module):
    def __init__(self,in_dim,activation):
        super(Self_Attn_Discriminator,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

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
        
        out = out + x
        return out,attention


class Generator(nn.Module):
    def __init__(self, pretrained_pgan, use_gpu=True):
        super(Generator, self).__init__()
        self.use_gpu = use_gpu

        # freeze generator parameters
        #       
        for param in pretrained_pgan.netG.parameters():
            param.requires_grad = False
        
            
        self.netG = copy.deepcopy(pretrained_pgan.netG)

        if use_gpu:
            self.netG.cuda()

        self.attn = Self_Attn_Generator(self.netG.dimOutput, 'relu')
    
    def forward(self, x):
        x = self.netG(x)
        x = self.attn(x)
        return x

    
class Discriminator(nn.Module):
    def __init__(self, pretrained_pgan, use_gpu=True):
        super(Discriminator, self).__init__()
        self.use_gpu = use_gpu

        # freeze discriminator parameters
        for param in pretrained_pgan.netD.parameters():
            param.requires_grad = False
        
        pretrained_pgan.netD.decisionLayer = nn.Identity() #Self_Attn(pretrained_pgan.netD.scalesDepth[0], 'relu')
                  
        self.netD = copy.deepcopy(pretrained_pgan.netD)
        if use_gpu:
            self.netD.cuda()
        
        self.attn = Self_Attn_Discriminator(self.netD.scalesDepth[0], 'relu')

        self.decisionLayer = EqualizedLinear(self.netD.scalesDepth[0],
                                             1,
                                             equalized=self.netD.equalizedlR,
                                             initBiasToZero=self.netD.initBiasToZero)
    
    def forward(self, x):
        x = self.netD(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x, attn = self.attn(x)
        x = x.squeeze()
        x = self.decisionLayer(x) # ovo treba srediti
        return x, attn
    

