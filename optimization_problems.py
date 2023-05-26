from abc import abstractmethod
from copy import deepcopy
from typing import Dict

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision

from models import MLP
from utils import device, count_parameters

BASELINES = ['SGD',]  # will add more

class TorchOptimizationProblem:
    def __init__(self, 
                initial_lr: float, 
                baseline_args: Dict[str, Dict],
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
        for k in baseline_args.keys():
            assert k in BASELINES, '{} is not a valid or implemented baseline'.format(k)
        self.lr = initial_lr
        self.baseline_args = baseline_args
        self.problem_args = problem_args
        self.reset(seed=seed)
    
    @abstractmethod
    def reset(self, seed=None):
        """
        Resets the problem to initialization. Uses `seed` for dataset generation or model initialization.
        
        MAKES USE OF THE ARGS IN    `self.problem_args` and `self.lr`
        NEEDS TO ASSIGN             `self.model`, `self.opt`, `self.train_x`, `self.test_x`, `self.train_y`, `self.test_y`
        ALSO MAKE `self.sgd_baseline` an exact copy of `self.model`
        """
        self.model = self.opt = self.train_x = self.test_x = self.train_y = self.test_y = None
        self.baselines = {}
        self.baseline_opts = {}
        
        """
        below is an example of how to initialize the baselines
        """
        if 'SGD' in self.baseline_args:
            self.baselines['SGD'] = deepcopy(self.model)
            self.baseline_opts['SGD'] = torch.optim.SGD(self.baselines['SGD'].parameters(), **self.baseline_args['SGD'])
        raise NotImplementedError()
    
    @abstractmethod
    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        loss function of the optimization. must be differentiable
        """
        raise NotImplementedError()
    
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
        prev_grad = None
        grad = 0.
        
        self.model.train()
        for m in self.baselines.values():
            m.train()
        
        train_dl = DataLoader(TensorDataset(self.train_x, self.train_y), batch_size=batch_size, shuffle=True, drop_last=True)
        dl = iter(train_dl)
        for _ in range(num_iters):
            try: 
                x, y = next(dl)
            except StopIteration: 
                dl = iter(train_dl)
                x, y = next(dl)
            
            x = x.to(device); y = y.to(device)
        
            # GD updates
            self.opt.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.opt.step()
            
            for k, m in self.baselines.items():
                try:
                    self.baseline_opts[k].zero_grad()
                    loss = self.loss_fn(m(x), y)
                    loss.backward()
                    self.baseline_opts[k].step()
                except:
                    pass
            
            # compute gradients of our method's trajectory
            i = 0
            g = np.zeros(count_parameters(self.model))
            for p in self.model.parameters():
                l = p.numel()
                g[i: i + l] = p.grad.reshape(-1).detach().cpu().data.numpy()
                i += l
            
            if prev_grad is None:
                prev_grad = g
            else:
                prev_grad = grad
                grad = g
            
        # compute error over entire test dataset
        self.model.eval()
        pred = self.model(self.test_x)
        error = self.error_fn(pred, self.test_y)
        
        baseline_errors = {}
        for k, m in self.baselines.items():
            m.eval()
            pred = m(self.test_x)
            baseline_errors[k] = self.error_fn(pred, self.test_y)
        
        grad_lr = -np.dot(grad, prev_grad)
        grad_mag = np.dot(grad, grad)
        return error, baseline_errors, grad_lr, grad_mag


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
        
        # baselines
        self.baselines = {}
        self.baseline_opts = {}
        if 'SGD' in self.baseline_args:
            self.baselines['SGD'] = deepcopy(self.model)
            self.baseline_opts['SGD'] = torch.optim.SGD(self.baselines['SGD'].parameters(), **self.baseline_args['SGD'])
        
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
        
        # baselines
        self.baselines = {}
        self.baseline_opts = {}
        if 'SGD' in self.baseline_args:
            self.baselines['SGD'] = deepcopy(self.model)
            self.baseline_opts['SGD'] = torch.optim.SGD(self.baselines['SGD'].parameters(), **self.baseline_args['SGD'])
        
    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, targets)
    
    def error_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            num_incorrect = (torch.softmax(logits, dim=-1).argmax(dim=-1) != targets).cpu().sum().item()
            return num_incorrect / logits.shape[0]
    