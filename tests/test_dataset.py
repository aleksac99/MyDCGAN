import unittest
import os
from torchvision.datasets import MNIST, CelebA
from torch.utils.data import DataLoader
from torchvision.transforms import Pad, PILToTensor, Compose, ToPILImage, Resize
import matplotlib.pyplot as plt

class testMNIST(unittest.TestCase):

    def test_loading(self):

        return 1
    
        transform = PILToTensor()
        t2pil = ToPILImage()
        mnist_train = MNIST(
            root=os.path.join('dataset', 'mnist'),
            download=True,
            transform=transform,
            train=True)
        
        train_loader = DataLoader(mnist_train, 8, shuffle=True)
        
        for imgs, labels in train_loader:
            
            imgs = imgs.float() / 255.
            fig, axs = plt.subplots(2, 4)
            for i, (img, label) in enumerate(zip(imgs, labels)):
                axs[i//4, i%4].imshow(t2pil(img))
                axs[i//4, i%4].set_title(label.item())

            # fig.savefig('test.png')
            return 1
        
class TestCelebA(unittest.TestCase):

    def test_loading(self):
        
        transform = Compose([
            PILToTensor(),
            # Pad([20, 0], 127),
            Resize((64, 64))
        ])
        t2pil = ToPILImage()
        celeba_train = CelebA(os.path.join('data', 'CelebA'), 'train', transform=transform, download=False)

        train_loader = DataLoader(celeba_train, 8, shuffle=True)

        for imgs, _ in train_loader:
            
            imgs = imgs.float() / 255.
            fig, axs = plt.subplots(2, 4)
            for i, img in enumerate(imgs):
                axs[i//4, i%4].imshow(t2pil(img))
                # axs[i//4, i%4].set_title(label.item())

            fig.savefig('test.png')
            return 1