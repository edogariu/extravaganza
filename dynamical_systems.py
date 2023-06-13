from abc import abstractmethod
from typing import Dict, Callable, Tuple

import random
from scipy.optimize import fmin
from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
import cocoex

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import jax

from controller import FloatController
from models import MLP, CNN
from utils import device

class DynamicalSystem:
    @abstractmethod
    def __init__(self):
        self.stats = {}
        pass
    
    @abstractmethod
    def reset(self, seed: int=None):
        self._set_seed(seed)
        pass
    
    @abstractmethod
    def reset_episode(self, seed: int=None):
        self._set_seed(seed)
        pass
    
    @abstractmethod
    def interact(self, control: float) -> float:
        """
        returns cost
        """
        pass
    
    def _set_seed(self, seed: int):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            jax.random.PRNGKey(seed)
            torch.manual_seed(seed)
        pass

class LinearRegression(DynamicalSystem):
    """
    GD over linear regression problem on one of the following datasets: `['california', 'diabetes', 'generated']`.
    Costs are measured in MSE loss values, and each interaction is one GD update.
    """
    def __init__(self, 
                 make_optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer], 
                 dataset: str='diabetes',
                 eval_every: int=None,
                 probe_fns: Dict[str, Callable]={},  # should be functions of the form f(self, control) -> float, where `self` is the DynamicalSystem and `control` is the input to .interact()
                 seed: int=None):
        
        assert dataset in ['california', 'diabetes', 'generated']
        self._set_seed(seed)
        self.make_optimizer = make_optimizer
        self.eval_every = eval_every
        self.probe_fns = probe_fns

        if dataset == 'california':
            X, y = fetch_california_housing(data_home='data', download_if_missing=True, return_X_y=True)
        elif dataset == 'diabetes':
            X, y = load_diabetes(return_X_y=True)
        elif dataset == 'generated':
            generation_args = {'n_features': 100, 'n_informative': 30, 'n_samples': 1000, 'noise': 0.1}
            X, y = make_regression(**generation_args)
        else:
            raise NotImplementedError(dataset)
        train_x, val_x, train_y, val_y = train_test_split(X, y)
        self.train_x = torch.FloatTensor(train_x, device=device)
        self.train_y = torch.FloatTensor(train_y, device=device).reshape(-1, 1)
        self.val_x = torch.FloatTensor(val_x, device=device)
        self.val_y = torch.FloatTensor(val_y, device=device).reshape(-1, 1)
        
        self.reset()
        pass
    
    def interact(self, control: FloatController) -> float:
        # train step
        self.opt.zero_grad()
        train_loss = torch.nn.functional.mse_loss(self.model(self.train_x), self.train_y)
        train_loss.backward()
        train_loss = train_loss.item()
        self.last_loss = train_loss
        self.opt.step()
        
        # probe and cache desired quantities
        self.stats['train_losses'][self.t] = train_loss
        try:
            lr = self.opt.param_groups[0]['lr']
            momentum = self.opt.param_groups[0]['momentum']
            self.stats['lrs'][self.t] = lr if not hasattr(lr, 'item') else lr.item()
            self.stats['momenta'][self.t] = momentum if not hasattr(momentum, 'item') else momentum.item()
        except KeyError:
            pass
        if hasattr(self.opt, '_grad_lr'):
            self.stats['true_grads'][self.t] = self.opt._grad_lr
        if hasattr(self.opt, '_B'):
            self.stats['true_Bs'][self.t] = self.opt._B    
        for k, f in self.probe_fns.items():
            self.stats[k][self.t] = f(self, control)
            
        # eval if desired
        if self.eval_every is not None and self.t % self.eval_every == 0:
            self.model.eval()
            with torch.no_grad():
                val_loss = torch.nn.functional.mse_loss(self.model(self.val_x), self.val_y)
                self.stats['val_losses'][self.t] = val_loss
            self.model.train()
            
        self.t += 1
        return train_loss
    
    def reset_episode(self):
        super().reset()  # sets seed
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.model.train()    
        pass
    
    def reset(self):
        super().reset()  # sets seed
        
        self.model = torch.nn.Linear(self.train_x.shape[1], 1).float()
        self.opt = self.make_optimizer(self.model)
        self.model.train().to(device)
        
        # stats to keep track of
        self.t = 0
        self.stats = {'train_losses': {},
                      'val_losses': {},
                      'lrs': {},
                      'momenta': {},
                      'true_grads': {},
                      'true_Bs': {}}
        assert not any([k in self.stats for k in self.probe_fns.keys()])
        self.stats.update({k: {} for k in self.probe_fns.keys()})
        self.last_loss = 0           
        if hasattr(self.opt, 'add_closure_func'):
            self.opt.add_closure_func(lambda: self.last_loss)
        pass
        
class MNIST(DynamicalSystem):
    """
    SGD over classification on MNIST.
    Costs are measured in NLL loss values, and each interaction is one SGD update with the given batch size.
    """
    def __init__(self, 
                 make_optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer], 
                 model_type: str='MLP',
                 batch_size: int=128,
                 eval_every: int=None,
                 probe_fns: Dict[str, Callable]={},  # should be functions of the form f(self, control) -> float, where `self` is the DynamicalSystem and `control` is the input to .interact()
                 seed: int=None):
        
        assert model_type in ['MLP', 'CNN']
        self._set_seed(seed)
        self.make_optimizer = make_optimizer
        self.model_type = model_type
        self.eval_every = eval_every
        self.probe_fns = probe_fns

        self.train_dl = DataLoader(
            torchvision.datasets.MNIST('./data', train=True, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_dl = DataLoader(
            torchvision.datasets.MNIST('./data', train=False, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=False, drop_last=True)
        self.dl = iter(self.train_dl)
        self.reset()
        pass
    
    def interact(self, control: FloatController) -> float:
        # train step
        try: 
            x, y = next(self.dl)
        except StopIteration: 
            self.dl = iter(self.train_dl)
            x, y = next(self.dl)
        x = x.to(device); y = y.to(device)
        self.opt.zero_grad()
        train_loss = torch.nn.functional.nll_loss(self.model(x).softmax(dim=-1).log(), y)
        train_loss.backward()
        train_loss = train_loss.item()
        self.last_loss = train_loss
        self.opt.step()
        
        # probe and cache desired quantities
        self.stats['train_losses'][self.t] = train_loss
        try:
            lr = self.opt.param_groups[0]['lr']
            momentum = self.opt.param_groups[0]['momentum']
            self.stats['lrs'][self.t] = lr if not hasattr(lr, 'item') else lr.item()
            self.stats['momenta'][self.t] = momentum if not hasattr(momentum, 'item') else momentum.item()
        except KeyError:
            pass
        if hasattr(self.opt, '_grad_lr'):
            self.stats['true_grads'][self.t] = self.opt._grad_lr
        if hasattr(self.opt, '_B'):
            self.stats['true_Bs'][self.t] = self.opt._B 
        for k, f in self.probe_fns.items():
            self.stats[k][self.t] = f(self, control)
            
        # eval if desired
        if self.eval_every is not None and self.t % self.eval_every == 0:
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.
                for x, y in self.val_dl:
                    x = x.to(device); y = y.to(device)
                    val_loss += torch.nn.functional.nll_loss(self.model(x).softmax(dim=-1).log(), y)
                val_loss /= len(self.val_dl)
                self.stats['val_losses'][self.t] = val_loss
            self.model.train()
            
        self.t += 1
        return train_loss
    
    def reset_episode(self):
        super().reset()  # sets seed
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.model.train()      
        pass
    
    def reset(self):
        super().reset()  # sets seed
        
        if self.model_type == 'MLP':
            self.model = MLP(layer_dims=[int(28 * 28), 1000, 1000, 10]).float()
        elif self.model_type == 'CNN':
            self.model = CNN(input_shape=(28, 28), output_dim=10)
        else:
            raise NotImplementedError(self.model_type)
        self.opt = self.make_optimizer(self.model)
        self.model.train().to(device)   
        
        # stats to keep track of
        self.t = 0
        self.stats = {'train_losses': {},
                      'val_losses': {},
                      'lrs': {},
                      'momenta': {},
                      'true_grads': {},
                      'true_Bs': {}}
        assert not any([k in self.stats for k in self.probe_fns.keys()])
        self.stats.update({k: {} for k in self.probe_fns.keys()})
        self.last_loss = 0           
        if hasattr(self.opt, 'add_closure_func'):
            self.opt.add_closure_func(lambda: self.last_loss)
        pass

class COCO(DynamicalSystem):
    def __init__(self, 
                 index: int,  # index of the problem in the suite to use
                 u_index: int,  # index of which coordinate of the problem we will optimize
                 predict_differences: bool,
                 probe_fns: Dict[str, Callable]={},  # should be functions of the form f(self, control) -> float, where `self` is the DynamicalSystem and `control` is the input to .interact()
                ):
        suite_name = "bbob"
        suite = cocoex.Suite(suite_name, "", "")
        self.problem = suite[index]
        self.u_index = u_index
        self.predict_differences = predict_differences
        self.initial_x = self.problem.initial_solution_proposal()  # fixed init
        # self.initial_x = self.problem.lower_bounds + np.random.rand(self.problem.dimension) * (self.problem.upper_bounds - self.problem.lower_bounds)  # random init
        self.interval = (self.problem.lower_bounds[u_index], self.problem.upper_bounds[u_index])
        self.probe_fns = probe_fns
        
        self.reset()
        pass
    
    def reset(self, seed: int=None):
        super().reset(seed)
        
        self.x = self.initial_x.copy()
        
        # stats to keep track of
        self.t = 0
        self.stats = {'controls': {},
                      'objectives': {}}
        assert not any([k in self.stats for k in self.probe_fns.keys()])
        self.stats.update({k: {} for k in self.probe_fns.keys()})
        
        # find optimal control
        _n = 10000
        _test = np.tile(self.x, _n).reshape(_n, -1)
        _test[:, self.u_index] = np.linspace(*self.interval, _n)
        gt_controls = _test[:, self.u_index]
        gt_values = [self.problem(_t) for _t in _test]
        optimal_control = gt_controls[np.argmin(gt_values)]
        self.stats['optimal_control'] = {'value': optimal_control}
        self.stats['gt_controls'] = {'value': gt_controls}
        self.stats['gt_values'] = {'value': gt_values}
        
        # def func(x):
        #     vec = self.initial_x.copy()
        #     vec[self.u_index] = x
        #     return self.problem(vec)
        # self.stats['fmin_controls'] = {t: v[0] for t, v in enumerate(fmin(func, self.initial_x[self.u_index], retall=True, disp=False)[:15])}
        pass
    
    def get_init(self) -> Tuple[float, Tuple[float, float]]:  # returns initial value and desired interval
        return (self.initial_x[self.u_index], self.interval) if not self.predict_differences else (0, (-1, 1))
        
    def interact(self, control: FloatController):
        if hasattr(control, 'item'): 
            c = control.item() 
        else:
            c = control
            
        if self.predict_differences:
            self.x[self.u_index] += c
        else:
            self.x[self.u_index] = c
        
        obj = self.problem(self.x)
        
        # probe and cache desired quantities
        self.stats['controls'][self.t] = self.x[self.u_index]
        self.stats['objectives'][self.t] = obj
        for k, f in self.probe_fns.items():
            self.stats[k][self.t] = f(self, control)
        self.t += 1
        return float(obj)
    
        
class SimpleSystem(DynamicalSystem):
    def __init__(self, 
                 problem_fn,
                 grad_fn,
                 controller_args,
                 predict_differences: bool,
                 use_grad: bool,
                 use_B: bool,
                 probe_fns: Dict[str, Callable]={},  # should be functions of the form f(self, control) -> float, where `self` is the DynamicalSystem and `control` is the input to .interact()
                 seed: int=None):
        
        self._set_seed(seed)
        self.predict_differences = predict_differences
        self.use_grad = use_grad
        self.use_B = use_B
        self.probe_fns = probe_fns
                
        self.problem_fn = problem_fn
        self.grad_fn = grad_fn
        self.B_fn = lambda x, prev_x: (self.problem_fn(x) - self.problem_fn(prev_x)) / (x - prev_x) if self.predict_differences and self.x != prev_x else 0
            
        self.initial_x = 0
        
        controller_args['initial_value'] = 0 if self.predict_differences else self.initial_x
        controller_args['bounds'] = (-0.1, 0.1) if self.predict_differences else (-10, 10)
        self.controller_args = controller_args
        self.reset()
        pass
    
    def interact(self, control: FloatController) -> float:
        u = self.controller.item() 
        if self.predict_differences:
            prev_x = self.x
            self.x += u
        else:
            prev_x = self.x = u
        
        obj = self.problem_fn(self.x)
        grad_u = self.grad_fn(self.x)
        B = self.B_fn(self.x, prev_x)
        self.controller.step(obj=obj, grad_u=grad_u if self.use_grad else None, B=B if self.use_B else None)
        
        self.stats['controls'][self.t] = self.x
        self.stats['objectives'][self.t] = obj
        self.stats['true_grads'][self.t] = grad_u
        self.stats['true_Bs'][self.t] = B
        for k, f in self.probe_fns.items():
            self.stats[k][self.t] = f(self, control)
        self.t += 1
        return obj
    
    def reset_episode(self):
        super().reset()  # sets seed
        pass
    
    def reset(self):
        super().reset()  # sets seed
        
        self.x = self.initial_x
        self.controller = FloatController(**self.controller_args)
        
        # stats to keep track of
        self.t = 0
        self.stats = {'objectives': {},
                      'controls': {},
                      'true_grads': {},
                      'true_Bs': {}}
        assert not any([k in self.stats for k in self.probe_fns.keys()])
        self.stats.update({k: {} for k in self.probe_fns.keys()})
        
        _n = 100000
        gt_controls = np.linspace(-10, 10, _n)
        gt_values = [self.problem_fn(u) for u in gt_controls]
        self.stats['optimal_control'] = {'value': gt_controls[np.argmin(gt_values)]}
        self.stats['gt_controls'] = {'value': gt_controls}
        self.stats['gt_values'] = {'value': gt_values}
        pass
    