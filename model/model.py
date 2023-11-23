import torch
import torch.nn as nn
import torch.nn.functional as F

class DCGANDiscriminator(nn.Module):

    def __init__(self, img_size, n_channels, n_filters, n_layers, l_relu_slope, kernel_size=3) -> None:

        super().__init__()

        # Params
        self.img_size = img_size
        self.n_layers = n_layers
        self.l_relu_slope = l_relu_slope
        self.n_filters = n_filters

        # Layers
        self.conv_layers = nn.ModuleList()
        tmp_img_size = img_size

        for i in range(n_layers):

            self.conv_layers.append(
                nn.Conv2d(
                    n_channels,
                    n_filters,
                    kernel_size,
                    stride=2,
                    padding=1))
            
            n_channels = n_filters
            n_filters *= 2
            tmp_img_size = (tmp_img_size + 2 - kernel_size) // 2 + 1

        n_out_features = tmp_img_size * tmp_img_size * n_channels
        self.flatten = nn.Flatten()
        self.out = nn.Linear(n_out_features, 1)



    def forward(self, x):
        
        for conv_layer in self.conv_layers:
            x = F.leaky_relu(conv_layer(x), negative_slope=self.l_relu_slope)
        
        x = self.flatten(x)

        return self.out(x) # NOTE: No activation in the final layer
    
class Reshape(nn.Module):

    def __init__(self, n_filters, img_size) -> None:
        super().__init__()

        self.n_filters = n_filters
        self.img_size = img_size

    def forward(self, x):

        return x.reshape(-1, self.n_filters, self.img_size, self.img_size)

class DCGANGenerator(nn.Module):

    def __init__(self, latent_dim, n_channels, n_filters, n_layers) -> None:

        super().__init__()

        self.latent_dim = latent_dim
        self.n_filters = n_filters

        self.dense = nn.Linear(latent_dim, n_filters*(2**n_layers) * 2 * 2) # NOTE: Hardcoded img_size
        self.reshape = Reshape(n_filters*(2**n_layers), 2)

        self.conv_t_layers = nn.ModuleList()
        tmp_filters = n_filters*(2**n_layers)
        for _ in range(n_layers):
            self.conv_t_layers.append(nn.ConvTranspose2d(tmp_filters, tmp_filters // 2, 4, 2, 1, bias=False))
            tmp_filters = tmp_filters // 2

        self.conv_t_layers.append(
            nn.ConvTranspose2d(tmp_filters, n_channels, kernel_size=1, stride=1, padding=2, bias=False))

    def forward(self, x):

        x = self.dense(x)
        x = self.reshape(x)
        for i, layer in enumerate(self.conv_t_layers):
            x = F.relu(layer(x)) if i!=len(self.conv_t_layers) else F.sigmoid(layer(x)) # NOTE: Put sigmoid instead of tanh to make outputs in range [0, 1]
        return x

