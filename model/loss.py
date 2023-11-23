import torch
import torch.nn as nn
from torch.autograd import grad

# from graphviz import Digraph
import torch
#from torchviz import make_dot



# TODO: Create Wasserstein Loss with Lipschitz penalty:
# penalty = (torch.clamp(grads - 1., min=0, max=None) ** 2).mean()

class WassersteinGPLoss(nn.Module):

    def __init__(self, l: float, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.l = l

    def forward(self, y_real: torch.Tensor, y_fake: torch.Tensor, y_mix: torch.Tensor, x_mix: torch.Tensor):

        grads = grad(y_mix, x_mix, torch.ones_like(y_mix), create_graph=True, retain_graph=True)[0]
        grads = grads.view(y_real.shape[0], -1)
        l_val = (y_fake.squeeze(-1) - y_real.squeeze(-1) + self.l * ((torch.norm(grads, 2, dim=-1) - 1.) ** 2)).mean()
        return l_val
    
class GeneratorLoss(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_fake):
        return (-y_fake).mean()