from typing import Dict, Tuple, Callable

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

from testing.base_problem import BaseTorchProblem
from testing.utils import top_1_accuracy

class TorchLinearRegression(BaseTorchProblem):
    def __init__(self, 
                 opt_args: Dict[str, Dict],
                 batch_size: int, 
                 n_features: int,
                 n_informative: int, 
                 n_samples: int, 
                 noise: float,
                 seed: int = None):
        """
        linear regression problem in PyTorch
        
        Parameters
        ----------
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
        """
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_samples = n_samples
        self.noise = noise
        super().__init__(opt_args, seed)
        
    def get_model(self, seed: int=None) -> torch.nn.Module:
        if seed is not None: torch.manual_seed(seed)
        return torch.nn.Linear(self.n_features, 1).float()
        
    def get_dataset(self, seed: int=None) -> Tuple[DataLoader, DataLoader]:
        if seed is not None: np.random.seed(seed)
            
        X, y = make_regression(n_samples=self.n_samples, n_features=self.n_features, n_informative=self.n_informative, noise=self.noise, random_state=seed)
        train_x, test_x, train_y, test_y = train_test_split(X, y)
        train_x, test_x, train_y, test_y = torch.FloatTensor(train_x), torch.FloatTensor(test_x), torch.FloatTensor(train_y).reshape(-1, 1), torch.FloatTensor(test_y).reshape(-1, 1)
        train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_dl = DataLoader(TensorDataset(test_x, test_y), batch_size=len(test_x), shuffle=False, drop_last=True)
        return train_dl, test_dl

    def loss_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(pred, targets)  # pred are in logits here
    
    def error_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> float:
        return self.loss_fn(pred, targets).item()
    
    
    
    
class TorchMNIST(BaseTorchProblem):
    def __init__(self, 
                opt_args: Dict[str, Dict],
                make_model: Callable[[], torch.nn.Module],
                batch_size: int,
                seed: int = None):
        """
        MNIST problem in PyTorch
        
        Parameters
        ----------
        batch_size : int
            batch size of train dataloader
        """
        self.make_model = make_model
        self.batch_size = batch_size
        super().__init__(opt_args, seed)
        
    def get_model(self, seed: int=None) -> torch.nn.Module:
        if seed is not None: torch.manual_seed(seed)
        return self.make_model()
        
    def get_dataset(self, seed: int=None) -> Tuple[DataLoader, DataLoader]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        train = torchvision.datasets.MNIST(root='data', train=True, download=True)
        test = torchvision.datasets.MNIST(root='data', train=False, download=True)
        train_x = train.data.float() / 128. - 0.5
        test_x = test.data.float() / 128. - 0.5
        train_y = train.targets
        test_y = test.targets
        train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_dl = DataLoader(TensorDataset(test_x, test_y), batch_size=len(test_x), shuffle=False, drop_last=True)
        return train_dl, test_dl
    
    def loss_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(pred, targets)  # pred are in logits here
    
    def error_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> float:
        return top_1_accuracy(pred, targets)  # pred is in logits
