import os
import torch
import json
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import PILToTensor, ToPILImage, Compose, Resize
from torchvision.datasets import MNIST, CelebA

from my_dcgan.model.model import DCGANDiscriminator, DCGANGenerator
from my_dcgan.model.loss import WassersteinGPLoss, GeneratorLoss
from my_dcgan.trainer.trainer import Trainer

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config file', type=str)

    return parser.parse_args()


def main():

    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    if cfg.get('img_size') is None:
        cfg['img_size'] = 64 if cfg['dataset']=='CELEBA' else 28

    if cfg.get('n_channels') is None:
        cfg['n_channels'] = 3 if cfg['dataset']=='CELEBA' else 1

    if cfg.get('n_layers') is None:
        cfg['n_layers'] = 5 if cfg['dataset']=='CELEBA' else 4

    # # Constants
    # DATASET = 'CELEBA'
    # OUT_DIR = 'out_celeba'
    # IMG_SIZE = 64 # TODO: 28 for MNIST, 64 for CELEBA
    # N_CHANNELS = 3 # TODO: 1 for MNIST, 3 for CELEBA
    # N_FILTERS = 64
    # L_RELU = 0.2
    # LOSS_L = 10
    # LATENT_DIM = 100
    # N_LAYERS = 5 # 4 for mnist, 5 for celeba

    # BATCH_SIZE = 64
    # LR = 3e-4
    # BETA1 = 0.5
    # BETA2 = 0.999
    # TRAIN_GEN_EACH = 5
    # N_EPOCHS = 1
    # SAVE_EACH = 1

    cfg['ckpt_dir'] = os.path.join(cfg['out_dir'], cfg['state_dict'])
    #CKPT_DIR = None

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load models
    discriminator = DCGANDiscriminator(
        cfg['img_size'],
        cfg['n_channels'],
        cfg['n_filters'],
        cfg['n_layers'],
        cfg['l_relu']).to(device)
    
    generator = DCGANGenerator(
        cfg['latent_dim'],
        cfg['n_channels'],
        cfg['n_filters'],
        cfg['n_layers'],
        cfg['dataset']=='MNIST').to(device)

    # Load dataset
    t2pil = ToPILImage()
    if cfg['dataset']=='MNIST':

        transform = PILToTensor()

        train_dataset = MNIST(
            root=os.path.join('data', 'mnist'),
            download=False,
            transform=transform,
            train=True)
        
    elif cfg['dataset']=='CELEBA':

        transform = Compose([
            PILToTensor(),
            Resize((64, 64))
            ])
        
        train_dataset = CelebA(os.path.join('data', 'CelebA'), 'train', transform=transform, download=True)
    else:
        raise ValueError(f'Dataset is {cfg["dataset"]}. Should be one of: `CELEBA`, `MNIST`.')
    
    train_loader = DataLoader(train_dataset, cfg['batch_size'], shuffle=True)

    # Losses and optimizers
    disc_criterion = WassersteinGPLoss(cfg['loss_l'])
    gen_criterion = GeneratorLoss()
    disc_optimizer = Adam(discriminator.parameters(), cfg['lr'], [cfg['beta1'], cfg['beta2']])
    gen_optimizer = Adam(generator.parameters(), cfg['lr'], [cfg['beta1'], cfg['beta2']])

    # Trainer
    trainer = Trainer(cfg['latent_dim'],
                      discriminator, generator,
                      disc_criterion, gen_criterion,
                      disc_optimizer, gen_optimizer, None,
                      cfg['batch_size'], train_loader, cfg['train_gen_each'], cfg['save_each'], device, cfg['out_dir'], cfg['ckpt_dir'])
    
    trainer.train(n_epochs=cfg['n_epochs'])

if __name__=='__main__':
    main()