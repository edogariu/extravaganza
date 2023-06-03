from typing import Dict, Tuple, Callable

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

from testing.base_problem import BaseTorchProblem
from testing.models import MLP, CNN
from testing.utils import top_1_accuracy

class TorchLinearRegression(BaseTorchProblem):
    def __init__(self, 
                 make_optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 batch_size: int, 
                 n_features: int,
                 n_informative: int, 
                 n_samples: int, 
                 noise: float,
                 seed: int=None,
                 probe_fns: Dict[str, Callable]={}):
        """
        linear regression problem in PyTorch
        
        Parameters
        ----------
        make_optimizer : Callable[[torch.nn.Module], torch.optim.Optimizer]
            function that makes optimizer of desired type to train the input module
        batch_size : int
            batch size of train dataloader
        n_features : int
            number of features to use
        n_informative : int
            number of informative features to use
        n_samples : int
            number of samples to make
        noise : float
            std of Gaussian noise to add to data points
        seed : int
            random seed
        probe_fns : Dict[str, Callable]
            functions that takes as input the `BaseTorchProblem` object and return probed values
        """
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_samples = n_samples
        self.noise = noise
        
        model = self.get_model(seed=seed)
        opt = make_optimizer(model)
        
        super().__init__(model, opt, seed=seed, probe_fns=probe_fns)
        # self.loss_fn = self.error_fn = torch.nn.functional.mse_loss 
        self.loss_fn = self.error_fn = lambda preds, targets: torch.nn.functional.mse_loss(preds, targets)
        
    def get_model(self, seed: int=None) -> torch.nn.Module:
        if seed is not None: 
            torch.manual_seed(seed)
        return torch.nn.Linear(self.n_features, 1).float()
        
    def get_dataset(self, seed: int=None) -> Tuple[DataLoader, DataLoader]:
        if seed is not None: 
            np.random.seed(seed)
        np.random.seed(0)
        X, y = make_regression(n_samples=self.n_samples, n_features=self.n_features, n_informative=self.n_informative, noise=self.noise, random_state=seed)
        train_x, val_x, train_y, val_y = train_test_split(X, y)
        train_x, val_x, train_y, val_y = torch.FloatTensor(train_x), torch.FloatTensor(val_x), torch.FloatTensor(train_y).reshape(-1, 1), torch.FloatTensor(val_y).reshape(-1, 1)
        train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dl = DataLoader(TensorDataset(val_x, val_y), batch_size=len(val_x), shuffle=False, drop_last=True)
        return train_dl, val_dl
    
    
class TorchMNIST(BaseTorchProblem):
    def __init__(self, 
                make_optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
                batch_size: int,
                model_type: str,
                seed: int=None,
                probe_fns: Dict[str, Callable]={}):
        """
        MNIST problem in PyTorch
        
        Parameters
        ----------
        make_optimizer : Callable[[torch.nn.Module], torch.optim.Optimizer]
            function that makes optimizer of desired type to train the input module
        batch_size : int
            batch size of train dataloader
        model_type : str
            must be one of `['mlp', 'cnn']`
        seed : int
            random seed
        probe_fns : Dict[str, Callable]
            functions that takes as input the `BaseTorchProblem` object and return probed values
        """
        assert model_type in ['mlp', 'cnn']
        self.batch_size = batch_size
        model = self.get_model(model_type, seed=seed)
        opt = make_optimizer(model)
        super().__init__(model, opt, seed=seed, probe_fns=probe_fns)
        
        # preds for the loss and error are in logits, not probs!!
        # self.loss_fn = torch.nn.functional.cross_entropy
        # self.error_fn = top_1_accuracy
 
        self.loss_fn = lambda logits, targets : torch.nn.functional.nll_loss(torch.softmax(logits, dim=-1).log(), targets)
        self.error_fn = lambda logits, targets : torch.nn.functional.nll_loss(torch.softmax(logits, dim=-1).log(), targets)
        
    def get_model(self, model_type: str, seed: int=None) -> torch.nn.Module:
        if seed is not None: 
            torch.manual_seed(seed)
        if model_type == 'mlp':
            model = MLP(layer_dims=[int(28 * 28), 1000, 1000, 10]).float()
        elif model_type == 'cnn':
            model = CNN(input_shape=(28, 28), output_dim=10)
        else:
            raise NotImplementedError(model_type)
        return model.float()
        
        
    def get_dataset(self, seed: int=None) -> Tuple[DataLoader, DataLoader]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        train_dl = DataLoader(
            torchvision.datasets.MNIST('./data', train=True, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dl = DataLoader(
            torchvision.datasets.MNIST('./data', train=False, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        # train = torchvision.datasets.MNIST(root='data', train=True, download=True)
        # val = torchvision.datasets.MNIST(root='data', train=False, download=True)
        # train_x = train.data.float() / 128. - 0.5
        # val_x = val.data.float() / 128. - 0.5
        # train_y = train.targets
        # val_y = val.targets
        # train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        # val_dl = DataLoader(TensorDataset(val_x, val_y), batch_size=len(val_x), shuffle=False, drop_last=True)
        return train_dl, val_dl

PROBLEM_CLASSES = {
    'LR': TorchLinearRegression, 
    'MNIST MLP': TorchMNIST,
    'MNIST CNN': TorchMNIST
}

PROBLEM_ARGS = {
    'LR': {
        'batch_size': 1,
        'n_features': 100,
        'n_informative': 30,
        'n_samples': 1000,
        'noise': 0.1,  # std dev of noise
        }, 
    'MNIST MLP': {
        'batch_size': 256,
        'model_type': 'mlp',
        },
    'MNIST CNN': {
        'batch_size': 64,
        'model_type': 'cnn',
        }
}
