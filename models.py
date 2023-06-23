from typing import List, Iterable

import torch
import torch.nn as nn

from utils import set_seed

class MLP(nn.Module):
    def __init__(self, 
                 layer_dims: List[int], 
                 activation: nn.Module = nn.ReLU,
                 normalization: nn.Module = nn.Identity,  # normalize before the activation
                 drop_last_activation: bool = True,
                 use_bias = True,
                 seed: int = None):
        """
        Creates a MLP to use as a weak learner

        Parameters
        ----------
        layer_dims : List[int]
            dimensions of each layer (`layer_dims[0]` should be input dim and `layer_dims[-1]` should be output dim)
        activation : nn.Module, optional
            activation function, by default nn.ReLU
        """
        
        super(MLP, self).__init__()
        set_seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        
        self.layers = [nn.Flatten(1)]
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i: i + 2]
            self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            self.layers.append(normalization(out_dim))
            self.layers.append(activation())
        if drop_last_activation: self.layers.pop()  # removes activation from final layer
        
        self.model = nn.Sequential(*self.layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model(x)
        return h
    
    
class CNN(nn.Module):
    def __init__(self, 
                 input_shape: Iterable,  # should be in CxHxW
                 output_dim: int,
                 activation: nn.Module=nn.ReLU,
                 use_bias=True,
                 seed: int = None):
        super(CNN, self).__init__()
        set_seed(seed)
        
        assert len(input_shape) in [2, 3]
        if len(input_shape) == 2: input_shape = (1, *input_shape)
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        self.body = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3),
            activation(),
            # nn.Conv2d(16, 32, kernel_size=3),
            # activation(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size(), 32, bias=use_bias),
            activation(),
            nn.Linear(32, self.output_dim, bias=use_bias)
        )
        
    def forward(self, x) -> torch.Tensor:
        x = x.reshape(-1, *self.input_shape)
        h = self.body(x)
        h = self.fc(h)
        return h
    
    def feature_size(self) -> int:
        with torch.no_grad():
            feature_size = self.body(torch.zeros(1, *self.input_shape)).view(1, -1).shape[1]
        return feature_size

if __name__ == '__main__':
    print('testing model dimension stuff!')
    _mlp = MLP([128, 256, 128, 64, 16])
    _mlp_test = torch.zeros((1, 128))
    assert _mlp(_mlp_test).shape[1] == 16
    
    _cnn = CNN(input_shape=(3, 210, 160), output_dim=16)
    _cnn_test = torch.zeros((1, 3, 210, 160))
    assert _cnn(_cnn_test).shape[1] == 16
    print('yippee!')
    
    from testing.utils import count_parameters
    mlp = MLP(layer_dims=[int(28 * 28), 100, 100, 100, 100, 10]).float()
    cnn = CNN(input_shape=(28, 28), output_dim=10)
    print(count_parameters(mlp), count_parameters(cnn))
