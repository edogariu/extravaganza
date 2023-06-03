from abc import abstractmethod
from typing import Dict, Tuple, Callable
import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from testing.utils import device

class BaseTorchProblem:
    """
    SUBCLASSES MUST IMPLEMENT 
    - `get_dataset(self, seed: int=None) -> Tuple[DataLoader, DataLoader]`
    - `loss_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`
    - `error_fn(self, pred: torch.Tensor, targets: torch.Tensor) -> float`
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 seed: int=None,
                 probe_fns: Dict[str, Callable]={}
                 ):
        """
        creates an optimization problem
        
        Parameters
        ----------
        model : torch.nn.Module
            model to train
        optimizer : torch.optim.Optimizer
            optimizer to train `model.parameters()`
        seed : int
            random seed
        probe_fns : Dict[str, Callable]
            functions that takes as input the `BaseTorchProblem` object and return probed values
        """
        self.model = model
        self.opt = optimizer
        self.reset(seed=seed)
        self.probe_fns = probe_fns
        
        self.t = 0
        self.last_loss = 0
        self.stats = {'train_losses': {},
                      'val_errors': {},
                      'lrs': {},
                      'momenta': {}}
        for k in self.probe_fns.keys():
            self.stats[k] = {}
            
        if hasattr(self.opt, 'add_closure_func'):
            self.opt.add_closure_func(lambda: self.last_loss)
        
    @abstractmethod
    def get_dataset(self, seed: int=None) -> Tuple[DataLoader, DataLoader]:
        """
        creates a new copy of the dataset for this problem. returns a train and val dataloader

        Parameters
        ----------
        seed : int, optional
            seed, by default None
        """
        train_dl = val_dl = ...
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
        self.train_dl, self.val_dl = self.get_dataset(seed)
        self.dl = iter(self.train_dl)
        
        # MODEL REINIT
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.model.train()
        pass
    
    def train_step(self):
        if not self.model.training: self.model.train()
        try: 
            x, y = next(self.dl)
        except StopIteration: 
            self.dl = iter(self.train_dl)
            x, y = next(self.dl)
        
        x = x.to(device); y = y.to(device)
        self.opt.zero_grad()
        loss = self.loss_fn(self.model(x), y)
        loss.backward()
        loss = loss.item()
        self.last_loss = loss
        self.opt.step()
        
        # probe and cache desired quantities
        self.stats['train_losses'][self.t] = loss
        lr = self.opt.param_groups[0]['lr']
        self.stats['lrs'][self.t] = lr if not hasattr(lr, 'item') else lr.item()
        try:
            momentum = self.opt.param_groups[0]['momentum']
            self.stats['momenta'][self.t] = momentum if not hasattr(momentum, 'item') else momentum.item()
        except KeyError:
            pass
        for k, f in self.probe_fns.items():
            self.stats[k][self.t] = f(self)
            
        self.t += 1
        return loss
    
    def eval(self):
        self.model.eval()
        errs = []
        with torch.no_grad():
            for x, y in self.val_dl:
                x = x.to(device); y = y.to(device)
                errs.append(self.error_fn(self.model(x), y))
            error = np.mean(errs)
            self.stats['val_errors'][self.t] = error
            self.model.train()
            return error

    def train(self, 
              num_iters: int, 
              eval_every: int, 
              reset_every: int,
              wordy: bool=False) -> Dict[str, Dict[int, float]]:
        """
        Trains for `num_iters` with given parameters.
        Returns the stats of the training
        
        Parameters
        ----------
        num_iters : int
            number of iterations to train
        eval_every : int
            how many iterations between each eval epoch on `self.val_dl`
        reset_every : int
            how many iterations between each training episode
        wordy : bool
            whether to use a progress bar
        """
        pbar = tqdm.trange(num_iters) if wordy else range(num_iters)
        for t in pbar:
            if t % reset_every == 0:
                self.reset()
            self.train_step()
            if t % eval_every == 0:
                self.eval()
            if wordy: pbar.set_postfix({'lr': self.stats['lrs'][t], 'loss': self.last_loss})
        
        return self.stats
