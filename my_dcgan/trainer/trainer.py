import os
import torch
from tqdm import tqdm
from torchvision.transforms import ToPILImage

import torch

class Trainer:

    def __init__(self, latent_dim, discriminator, generator, disc_criterion, gen_criterion, disc_optimizer, gen_optimizer, lr_scheduler, gen_batch_size, train_loader, train_gen_each, save_each, device, save_dir, ckpt_path) -> None:

        self.latent_dim = latent_dim
        self.discriminator = discriminator
        self.generator = generator
        self.disc_criterion = disc_criterion
        self.gen_criterion = gen_criterion
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.train_gen_each = train_gen_each
        self.save_each = save_each
        self.device = device
        self.gen_batch_size = gen_batch_size
        self.save_dir = save_dir

        self.disc_loss = []
        self.gen_loss = []

        self.starting_epoch = self.try_load_from_ckpt(ckpt_path)

    def _train_epoch(self, pbar):

        running_disc_loss = 0.
        running_gen_loss = 0.
        for batch_idx, (x_real, _) in enumerate(self.train_loader):

            x_real = x_real.float() / 255.
            running_disc_loss += self._train_disc(x_real)

            if ((batch_idx + 1) % self.train_gen_each) == 0:
                running_gen_loss += self._train_gen()

            pbar.set_description(f'Epoch progress: {int(batch_idx/len(self.train_loader)*100)}%')

        return running_disc_loss / len(self.train_loader), running_gen_loss / (len(self.train_loader) // self.train_gen_each)

    def _train_disc(self, x_real):

        # Freeze generator
        for param in self.generator.parameters():
            param.requires_grad = False
        # Unfreeze discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = True

        batch_size = x_real.shape[0]
        z = -2. * torch.rand(batch_size, self.latent_dim) + 1.

        # Move to device
        x_real = x_real.to(self.device)
        z = z.to(self.device)

        x_fake = self.generator(z)

        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)

        eta = torch.rand(batch_size, 1, 1, 1).to(self.device)
        x_mix = eta * x_real + (1-eta) * x_fake
        x_mix.requires_grad = True
        y_mix = self.discriminator(x_mix)

        loss = self.disc_criterion(y_real, y_fake, y_mix, x_mix)

        self.disc_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()
        loss.backward()

        self.disc_optimizer.step()

        return loss.item()

    def _train_gen(self):

        # Freeze discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = False
        # Unfreeze generator
        for param in self.generator.parameters():
            param.requires_grad = True

        # Sample from normal dist
        z = -2. * torch.rand(self.gen_batch_size, self.latent_dim) + 1.
        z = z.to(self.device)
        x_fake = self.generator(z)
        y_fake = self.discriminator(x_fake)

        loss = self.gen_criterion(y_fake)

        self.disc_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()

        loss.backward()

        self.gen_optimizer.step()

        return loss.item()

    def train(self, n_epochs):

        z_fixed = -2. * torch.rand(36, self.latent_dim) + 1.
        z_fixed = z_fixed.to(self.device)
        pbar = tqdm(range(self.starting_epoch, self.starting_epoch + n_epochs), leave=True)

        for epoch in pbar:

            disc_loss, gen_loss = self._train_epoch(pbar)

            self.disc_loss.append(disc_loss)
            self.gen_loss.append(gen_loss)

            
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, 'losses.txt'), 'w') as f:
                data = '\n'.join(
                    [','.join([f'{dl:.3f}', f'{gl:.3f}'])
                        for dl, gl in zip(self.disc_loss, self.gen_loss)])

                f.write(data)

            if (epoch + 1) % self.save_each == 0:

                self.save_generated_samples(epoch + 1, z_fixed)
            self.save_ckpt(epoch + 1, self.disc_loss, self.gen_loss)

        # Save final model and samples
        self.save_generated_samples(self.starting_epoch + n_epochs, z_fixed)
        self.save_ckpt(self.starting_epoch + n_epochs,
                       self.disc_loss, self.gen_loss)
        print(
            f'Final checkpoint saved at epoch {self.starting_epoch + n_epochs}')

    def save_generated_samples(self, epoch, z):

        with torch.no_grad():

            x_fake = self.generator(z)

            t2pil = ToPILImage()
            os.makedirs(os.path.join(self.save_dir, str(epoch)), exist_ok=True)
            for idx, out in enumerate(x_fake):
                img = t2pil(out)
                img.save(os.path.join(self.save_dir, str(epoch), f'{str(idx)}.png'))


    def save_ckpt(self, epoch, disc_loss, gen_loss):

        torch.save({
            'epoch': epoch,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_loss': disc_loss,
            'gen_loss': gen_loss, }, os.path.join(self.save_dir, 'state_dict.pt'))

    def try_load_from_ckpt(self, ckpt_path):

        if isinstance(ckpt_path, str) and os.path.exists(ckpt_path):

            ckpt = torch.load(ckpt_path)

            self.discriminator.load_state_dict(
                ckpt['discriminator_state_dict'])
            self.generator.load_state_dict(ckpt['generator_state_dict'])
            self.disc_optimizer.load_state_dict(
                ckpt['disc_optimizer_state_dict'])
            self.gen_optimizer.load_state_dict(
                ckpt['gen_optimizer_state_dict'])
            self.disc_loss = ckpt['disc_loss']
            self.gen_loss = ckpt['gen_loss']
            epoch = ckpt['epoch']

            print(
                f'Successfully loaded checkpoint. Training starting from epoch {epoch + 1}')
            return epoch
        print('Checkpoint not found. Starting from scratch.')
        return 0
