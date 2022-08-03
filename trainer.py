import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from model import Generator, Discriminator
from utils import *
import wandb


class Trainer(object):
    def __init__(self, data_loader, config, pretrained_model):
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.sigma_update_step = config.sigma_update_step
        self.sigma_update_delta = config.sigma_update_delta
        self.accum_step = config.accum_step

        self.version = config.version
        self.use_gpu = config.use_gpu

        self.pretrained_pgan = pretrained_model

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()
        wandb.init(entity = 'pavlepadjin', project = 'image-generation-gan')
        wandb.watch_called = False

        wconfig = wandb.config          # Initialize config
        wconfig.batch_size = self.batch_size          # input batch size for training (default: 64)
        wconfig.g_lr = self.g_lr 
        wconfig.d_lr = self.d_lr
        wconfig.momentum = 0.1          # SGD momentum (default: 0.5) 
        wconfig.cuda  = self.use_gpu         # disables CUDA training
    

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        #self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        #self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
        self.G = Generator(pretrained_pgan=self.pretrained_pgan, use_gpu = self.use_gpu)
        self.G.to(self.device)

        self.D = Discriminator(pretrained_pgan=self.pretrained_pgan, use_gpu = self.use_gpu)
        self.D.to(self.device)

        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        

        
    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))


    def train(self):

        wandb.watch(self.G, log="all")
        wandb.watch(self.D, log="all")


        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = self.model_save_step #int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
    
        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        start_time = time.time()
        g_loss_log = []
        example_images = []
        for step in range(start, self.total_step):
            
             # ================== Train D ================== #
            self.D.train()
            self.G.train()


            try:
                real_images = next(data_iter) # real_images, _ = next(data_iter)
            except: # if we went through the whole dataset, start again
                data_iter = iter(self.data_loader)
                real_images = next(data_iter) #real_images, _ = next(data_iter)
            
            real_images = tensor2var(real_images).to(self.device)

             # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            d_out_real = self.D(real_images)
            
            d_loss_real = - torch.mean(d_out_real)
           
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images, gf = self.G(z)
            d_out_fake = self.D(fake_images)

            d_loss_fake = d_out_fake.mean()

            d_loss = d_loss_real + d_loss_fake

            d_loss.backward()
            
            if (step + 1) % self.accum_step == 0:
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()


            # Compute gradient penalty
            alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device)
            
            alpha = alpha.expand_as(real_images)
            interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
            out = self.D(interpolated)

            grad_outputs = torch.ones(out.size()).to(self.device)

            grad = torch.autograd.grad(outputs=out,
                                        inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            d_loss = self.lambda_gp * d_loss_gp

            
        
            # ================== Train G ================== #

             # Create random noise
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,_ = self.G(z)

            # Compute loss with fake images
            g_out_fake = self.D(fake_images)  # batch x n
           
            g_loss_fake = - g_out_fake.mean()

            d_loss.backward()
            g_loss_fake.backward()

            if (step + 1) % self.accum_step == 0:
                self.d_optimizer.step()
                self.g_optimizer.step()
                self.reset_grad()
           
            


            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_real.data.item(),
                             self.G.attn.attn_base.gamma.mean().data.item()))

            # Update sigma of attention layers
            if (step + 1) % self.sigma_update_step == 0:
                self.G.attn.update_sigma(self.G.attn.sigma + self.sigma_update_delta)
                self.D.attn.update_sigma(self.D.attn.sigma + self.sigma_update_delta)
                print('Updated sigma to {}'.format(self.G.attn.sigma))

        

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_= self.G(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))
                g_loss_log.append(g_loss_fake.data.item())
                example_images.append(denorm(fake_images.data))
                wandb.log({
                    "Examples":  [wandb.Image(denorm(fake_images.data), caption=f"Step {step + 1}")],
                    "Training G Loss": g_loss_fake.data.item(),
                    "Training D Loss": d_loss.data.item(),
                    "Training Gamma": self.G.attn.attn_base.gamma.mean().data.item(),
                    "Training D loss real": d_loss_real.data.item(),
                    "Generator attention" : [wandb.Image(gf)]})

            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

