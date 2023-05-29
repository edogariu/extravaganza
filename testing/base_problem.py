from abc import abstractmethod
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from testing.utils import device, count_parameters

METHODS = ['ours', 'SGD',]  # will add more when i add them to .reset()

class BaseTorchProblem:
    """
    SUBCLASSES MUST IMPLEMENT 
    - `get_model(self, seed: int=None) -> torch.nn.Module`
    - `get_dataset(self, seed: int=None) -> Tuple[DataLoader, DataLoader]`
    - `loss_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`
    - `error_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> float`
    """
    def __init__(self, 
                opt_args: Dict[str, Dict],
                seed: int=None
                ):
        """
        creates an optimization problem
        """
        for k in opt_args.keys():
            assert k in METHODS, '{} is not a valid or implemented optimization method'.format(k)
            
        self.opt_args = opt_args
        self.reset(seed=seed)
        
    @abstractmethod
    def get_model(self, seed: int=None) -> torch.nn.Module:
        """
        returns a newly initialized version of the model we will use
        """
        model = ...
        raise NotImplementedError()
        
    @abstractmethod
    def get_dataset(self, seed: int=None) -> Tuple[DataLoader, DataLoader]:
        """
        creates a new copy of the dataset for this problem. returns a train and test dataloader

        Parameters
        ----------
        seed : int, optional
            seed, by default None
        """
        train_dl = test_dl = ...
        raise NotImplementedError()
    
    @abstractmethod
    def loss_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = ...
        raise NotImplementedError()
        
    @abstractmethod
    def error_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> float:
        error = ...
        raise NotImplementedError()
        
    
    def reset(self, seed=None):
        """
        Resets the problem to initialization. Uses `seed` for dataset generation or model initialization.
        """
        # SEED
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # DATASET
        self.train_dl, self.test_dl = self.get_dataset(seed)
        
        # MODELS and OPTIMIZERS
        model = self.get_model()
        for layer in model.children():  # reinitialize the model
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        self.models = {}
        self.opts = {}
        for k, args in self.opt_args.items():
            m = deepcopy(model); m.load_state_dict(model.state_dict())
            self.models[k] = m
            
            """
            TODO HERE IS WHERE TO INITIALIZE NEW OPTIMIZERS
            """
            if k == 'ours':
                self.opts[k] = torch.optim.SGD(m.parameters(), **args)
            elif k == 'SGD':
                self.opts[k] = torch.optim.SGD(m.parameters(), **args)
        pass

    def train_step(self, 
                   x: torch.Tensor,
                   y: torch.Tensor) -> float:
        """
        steps the optimization of the main copy and the baselines once.
        returns the vector of gradient of loss function w.r.t the parameters of the model

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            outputs
        """
        for m in self.models.values():
            m.train()
        
        # GD updates
        for model, opt in zip(self.models.values(), self.opts.values()):
            opt.zero_grad()
            loss = self.loss_fn(model(x), y)
            loss.backward()
            opt.step()
    
        # assemble gradient vector for our method
        if 'ours' in self.models:
            i = 0
            grad = np.zeros(count_parameters(self.models['ours']))
            for p in self.models['ours'].parameters():
                l = p.numel()
                grad[i: i + l] = p.grad.reshape(-1).detach().cpu().data.numpy()
                i += l
            return grad
        else:
            return None
    
    def compute_error(self, dataloader: DataLoader) -> float:
        """
        computes the mean error over an entire dataset.

        Parameters
        ----------
        dataloader : DataLoader
            the dataset
        """
        for m in self.models.values():
            m.eval()
            
        errors = defaultdict(list)
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device); y = y.to(device)
                
                for k, m in self.models.items():
                    errors[k].append(self.error_fn(m(x), y))
                    
        errors = {k: np.mean(err) for k, err in errors.items()}  # take means
        return errors
    
    def train(self, num_iters: int) -> Dict[str, Any]:
        """
        trains for `num_iters` with given batch size
        
        Parameters
        ----------
        num_iters : int
            number of iterations to train
        """
        prev_grad = None
        grad = None
        
        dl = iter(self.train_dl)
        for _ in range(num_iters):
            try: 
                x, y = next(dl)
            except StopIteration: 
                dl = iter(self.train_dl)
                x, y = next(dl)
            
            x = x.to(device); y = y.to(device)
            g = self.train_step(x, y)
            
            prev_grad = grad if grad is not None else g
            grad = g
            
        # compute error over entire test dataset
        errors = self.compute_error(self.test_dl)
        ret = {'errors': errors}
        
        # compute gradients
        if grad is not None:
            ret['grad_lr'] = -np.dot(grad, prev_grad)
            ret['grad_mag'] = np.dot(grad, grad)

        return ret
