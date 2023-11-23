import torch
import unittest

from model.model import DCGANDiscriminator, DCGANGenerator


class TestDiscriminator(unittest.TestCase):

    def test_forward_pass(self):

        x = torch.randn(16, 1, 28, 28)

        discriminator = DCGANDiscriminator(28, 1, 32, 4, 10, 0.2, 3)

        y = discriminator(x)

        print(y.shape)


class TestGenerator(unittest.TestCase):

    def test_forward_pass(self):

        z = torch.randn(16, 100)
        generator = DCGANGenerator(100, 1, 32, 4)
        # print(generator)
        x = generator(z)
        print(x.shape)