from abc import abstractmethod

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision

from models import MLP
from utils import device

class TorchOptimizationProblem:
    def __init__(self, 
                initial_lr: float, 
                seed: int=None,
                **problem_args,
                ):
        """
        forms an optimization problem, using the args given in `problem_args`
        This can be used to get observations (and maybe even predictions) and also to see if updating the learning rates is helpful

        Parameters
        ----------
        initial_lr : float
            learning rate to start with
        seed : int
            random seed
        """
        self.lr = initial_lr
        self.problem_args = problem_args
        self.reset(seed=seed)
    
    @abstractmethod
    def reset(self, seed=None):
        """
        Resets the problem to initialization. Uses `seed` for dataset generation or model initialization.
        
        MAKES USE OF THE ARGS IN    `self.problem_args` and `self.lr`
        NEEDS TO ASSIGN             `self.model`, `self.opt`, `self.train_x`, `self.test_x`, `self.train_y`, `self.test_y`
        """
        self.model = self.opt = self.train_x = self.test_x = self.train_y = self.test_y = None
        pass
    
    @abstractmethod
    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        loss function of the optimization. must be differentiable
        """
        pass
    
    @abstractmethod
    def error_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        error function of the optimization. need not be differentiable
        """
        with torch.no_grad():
            return self.loss_fn(preds, targets).item()
    
    def train(self,
              num_iters: int,
              batch_size: int
              ) -> float:
        """
        returns final validation error when training with SGD for `num_iters` with given batch size
        
        Parameters
        ----------
        num_iters : int
            number of iterations to train
        batch_size : int
            batch size to train with
        """
        self.model.train()
        train_dl = DataLoader(TensorDataset(self.train_x, self.train_y), batch_size=batch_size, shuffle=True, drop_last=True)
        dl = iter(train_dl)
        for _ in range(num_iters):
            try: 
                x, y = next(dl)
            except StopIteration: 
                dl = iter(train_dl)
                x, y = next(dl)
            
            x = x.to(device); y = y.to(device)
        
            # GD update
            self.opt.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.opt.step()
            
        # compute error over entire test dataset
        self.model.eval()
        pred = self.model(self.test_x)
        error = self.error_fn(pred, self.test_y)
        return error


class TorchLinearRegression(TorchOptimizationProblem):
    def reset(self, seed=None):
        """
        Resets the linear regression to model initialization, with a new dataset.
        """
        # SEED
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # DATASET
        X, y = make_regression(**self.problem_args, random_state=seed)
        train_x, test_x, train_y, test_y = train_test_split(X, y)
        self.train_x, self.test_x, self.train_y, self.test_y = torch.FloatTensor(train_x), torch.FloatTensor(test_x), torch.FloatTensor(train_y).reshape(-1, 1), torch.FloatTensor(test_y).reshape(-1, 1)

        # MODEL
        self.model = torch.nn.Linear(self.train_x.shape[1], 1).train().float()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
    def loss_fn(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(inputs, targets)


class TorchMNISTMLP(TorchOptimizationProblem):
    """
    PyTorch MNIST optimization problem
    """
    def reset(self, seed=None):
        """
        Resets the linear regression to model initialization, with a new dataset.
        """
        # SEED
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # DATASET
        train = torchvision.datasets.MNIST(root='data', train=True, download=True)
        test = torchvision.datasets.MNIST(root='data', train=False, download=True)
        self.train_x = train.data.float() / 128. - 0.5
        self.test_x = test.data.float() / 128. - 0.5
        self.train_y = train.targets
        self.test_y = test.targets
        
        # MODEL
        self.model = MLP(**self.problem_args).train().float()  # predicts logits
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, targets)
    
    def error_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            num_incorrect = (torch.softmax(logits, dim=-1).argmax(dim=-1) != targets).cpu().sum().item()
            return num_incorrect / logits.shape[0]
    