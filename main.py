import os
import torch
from torch.utils.data import DataLoader #, Subset
from torch.optim import Adam
from torchvision.transforms import PILToTensor, ToPILImage
from torchvision.datasets import MNIST

from model.model import DCGANDiscriminator, DCGANGenerator
from model.loss import WassersteinGPLoss, GeneratorLoss
from trainer.trainer import Trainer


if __name__=='__main__':

    # Constants
    IMG_SIZE = 28
    N_CHANNELS = 1
    N_FILTERS = 64
    L_RELU = 0.2
    LOSS_L = 10
    LATENT_DIM = 100
    N_LAYERS = 4

    BATCH_SIZE = 64
    LR = 3e-4
    BETA1 = 0.5
    BETA2 = 0.999
    TRAIN_GEN_EACH = 5
    N_EPOCHS = 5
    SAVE_EACH = 1

    CKPT_DIR = 'output/state_dict.pt'
    #CKPT_DIR = None

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load models
    discriminator = DCGANDiscriminator(IMG_SIZE, N_CHANNELS, N_FILTERS, N_LAYERS, L_RELU).to(device)
    
    generator = DCGANGenerator(LATENT_DIM, N_CHANNELS, N_FILTERS, N_LAYERS).to(device)

    # Load dataset
    transform = PILToTensor()

    t2pil = ToPILImage()
    train_dataset = MNIST(
        root=os.path.join('dataset', 'mnist'),
        download=True,
        transform=transform,
        train=True)
    
    # train_dataset = Subset(train_dataset, range(1000))
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

    # Losses and optimizers
    disc_criterion = WassersteinGPLoss(LOSS_L)
    gen_criterion = GeneratorLoss()
    disc_optimizer = Adam(discriminator.parameters(), LR, [BETA1, BETA2])
    gen_optimizer = Adam(generator.parameters(), LR, [BETA1, BETA2])

    # Trainer
    trainer = Trainer(LATENT_DIM,
                      discriminator, generator,
                      disc_criterion, gen_criterion,
                      disc_optimizer, gen_optimizer, None,
                      BATCH_SIZE, train_loader, TRAIN_GEN_EACH, SAVE_EACH, device, 'output', CKPT_DIR)
    

    trainer.train(n_epochs=N_EPOCHS)