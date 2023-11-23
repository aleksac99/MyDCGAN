import unittest
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor, ToPILImage
import matplotlib.pyplot as plt

class testMNIST(unittest.TestCase):

    def test_loading(self):


        transform = PILToTensor()
        t2pil = ToPILImage()
        mnist_train = MNIST(
            root=os.path.join('dataset', 'mnist'),
            download=True,
            transform=transform,
            train=True)
        
        train_loader = DataLoader(mnist_train, 8, shuffle=True)
        
        for imgs, labels in train_loader:

            fig, axs = plt.subplots(2, 4)
            for i, (img, label) in enumerate(zip(imgs, labels)):
                axs[i//4, i%4].imshow(t2pil(img))
                axs[i//4, i%4].set_title(label.item())

            # fig.savefig('test.png')
            return 1