from parameters import *
from trainer import Trainer
import torch
from datasets import FaceDataset
from torch.backends import cudnn
from utils import make_folder
from torch.utils.data import DataLoader


def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    dataset = FaceDataset(config.image_path)

    data_loader = DataLoader(dataset, 
                             batch_size = config.batch_size, shuffle=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)

    pretrained_model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celeba',
                       pretrained=True, useGPU=True)

    if config.train:
        trainer = Trainer(data_loader, config, pretrained_model)

        trainer.train()
    else:
        pass
        #tester = Tester(data_loader.loader(), config)
        #tester.test()

if __name__ == '__main__': 
    config = get_parameters()
    print(config)
    main(config)