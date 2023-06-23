from abc import abstractmethod
from typing import Callable, Tuple, Union, Iterator
from collections import defaultdict

from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
import cocoex

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import jax
import jax.numpy as jnp

from models import MLP, CNN
from stats import Stats
from utils import device, set_seed, jkey, sample
        
class DynamicalSystem:
    
    @abstractmethod
    def __init__(self, seed: int = None):
        # needs to set the following things
        set_seed(seed)  # for reproducibility
        self.stats: Stats = None
        self.OBSERVABLE: bool = None
        raise NotImplementedError()
    
    @abstractmethod
    def reset(self, seed: int = None):
        """
        to reset an episode, which should send state back to init
        """
        set_seed(seed)  # for reproducibility
        return self
    
    @abstractmethod
    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        given control, returns cost and an observation. The observation may be the true state, a function of the state, or simply `None`
        """
        raise NotImplementedError()
    
    
class NNTraining(DynamicalSystem):
    """
    neural network training is a dynamical system with learning rate and momentum as controls
    """
    @abstractmethod
    def __init__(self, seed: int = None):
        set_seed(seed)  # for reproducibility
        
        # needs to set the following things
        self.stats: Stats = None
        self.OBSERVABLE: bool = False
        self.model: torch.nn.Module = None   # this is basically the state
        self.opt: torch.optim.Optimizer = None
        self.apply_control: Callable[[jnp.ndarray, DynamicalSystem],] = None  # applies the control to the system (which might update the learning rate or momentum or something)
        self.train_dl: DataLoader = None
        self.val_dl: DataLoader = None
        self.dl: Iterator = None
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # loss(yhat, y)
        self.eval_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # loss(yhat, y)
        self.eval_every: int = None
        self.t: int = 0
        
        self.reset(seed)
        raise NotImplementedError()
    
    def reset(self, seed: int = None):
        """
        to reset an episode, which should send state back to init
        """
        set_seed(seed)  # for reproducibility
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.model.train()  
        self.opt.__setstate__({'state': defaultdict(dict)})  
        return self
    
    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        # apply control
        self.apply_control(control, self)
        
        # train step (observe costs)
        try: 
            x, y = next(self.dl)
        except StopIteration: 
            self.dl = iter(self.train_dl)
            x, y = next(self.dl)
        x = x.to(device); y = y.to(device)
        self.opt.zero_grad()
        train_loss = self.loss_fn(self.model(x), y)
        train_loss.backward()
        self.opt.step()
        train_loss = train_loss.item()
        
        # update stats
        self.stats.update('train losses', train_loss, t=self.t)
        try:
            self.stats.update('lrs', self.opt.param_groups[0]['lr'], t=self.t)
            self.stats.update('momenta', float(self.opt.param_groups[0]['momentum']), t=self.t)
        except KeyError:
            pass
        
        # eval if desired
        if self.eval_every is not None and self.t % self.eval_every == 0:
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.
                for x, y in self.val_dl:
                    x = x.to(device); y = y.to(device)
                    val_loss += self.loss_fn(self.model(x), y)
                val_loss /= len(self.val_dl)
                self.stats.update('val losses', val_loss.item(), t=self.t)
            self.model.train()
            
        self.t += 1
        return train_loss, None

class LinearRegression(NNTraining):
    """
    GD over linear regression problem on one of the following datasets: `['california', 'diabetes', 'generated']`.
    Costs are measured in MSE loss values, and each interaction is one full GD update.
    """
    def __init__(self, 
                 make_optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer], 
                 apply_control: Callable[[jnp.ndarray, DynamicalSystem], None],
                 dataset: str = 'diabetes',
                 eval_every: int = None,
                 seed: int = None):
        
        set_seed(seed)  # for reproducibility

        self.eval_every = eval_every
        self.OBSERVABLE = False
        self.apply_control = apply_control

        # data
        assert dataset in ['california', 'diabetes', 'generated']
        if dataset == 'california':
            X, y = fetch_california_housing(data_home='data', download_if_missing=True, return_X_y=True)
        elif dataset == 'diabetes':
            X, y = load_diabetes(return_X_y=True)
        elif dataset == 'generated':
            generation_args = {'n_features': 100, 'n_informative': 30, 'n_samples': 1000, 'noise': 0.1}
            X, y = make_regression(**generation_args, random_state=seed)
        
        train_x, val_x, train_y, val_y = train_test_split(X, y)
        train_x = torch.FloatTensor(train_x, device=device)
        train_y = torch.FloatTensor(train_y, device=device).reshape(-1, 1)
        val_x = torch.FloatTensor(val_x, device=device)
        val_y = torch.FloatTensor(val_y, device=device).reshape(-1, 1)
        self.train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=train_x.shape[0], shuffle=False)
        self.val_dl = DataLoader(TensorDataset(val_x, val_y), batch_size=val_x.shape[0], shuffle=False)
        self.dl = iter(self.train_dl)
        
        # model
        self.model = torch.nn.Linear(train_x.shape[1], 1).float()
        self.opt = make_optimizer(self.model)
        self.model.train().to(device)
        
        # losses
        self.loss_fn = self.eval_fn = lambda yhat, y: torch.nn.functional.mse_loss(yhat, y)
        
        # stats to keep track of
        self.t = 0
        self.stats = Stats()
        self.stats.register('train losses', float, plottable=True)
        self.stats.register('val losses', float, plottable=True)
        self.stats.register('lrs', float, plottable=True)
        self.stats.register('momenta', float, plottable=True)
        
        self.reset(seed)  # sets random state for the beginning of the episode (in case it's changed during __init__)
        pass
        
class MNIST(NNTraining):
    """
    SGD over classification on MNIST.
    Costs are measured in NLL loss values, and each interaction is one SGD update with the given batch size.
    """
    def __init__(self, 
                 make_optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer], 
                 apply_control: Callable[[jnp.ndarray, DynamicalSystem], None],
                 model_type: str = 'MLP',
                 batch_size: int = 128,
                 eval_every: int = None,
                 seed: int = None):
        
        set_seed(seed)  # for reproducibility
       
        # needs to set the following things
        self.apply_control: Callable[[jnp.ndarray, DynamicalSystem],] = None  # applies the control to the system (which might update the learning rate or momentum or something)
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # loss(yhat, y)
        self.eval_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # loss(yhat, y)
        
        self.eval_every = eval_every
        self.OBSERVABLE = False
        self.apply_control = apply_control

        # data
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
        
        # model
        assert model_type in ['MLP', 'CNN']
        if model_type == 'MLP':
            self.model = MLP(layer_dims=[int(28 * 28), 1000, 1000, 10]).float()
        elif model_type == 'CNN':
            self.model = CNN(input_shape=(28, 28), output_dim=10)
        self.opt = make_optimizer(self.model)
        self.model.train().to(device)  
        
        # losses
        self.loss_fn = self.eval_fn = lambda yhat, y:  torch.nn.functional.nll_loss(yhat.softmax(dim=-1).log(), y)
        
        # stats to keep track of
        self.t = 0
        self.stats = Stats()
        self.stats.register('train losses', float, plottable=True)
        self.stats.register('val losses', float, plottable=True)
        self.stats.register('lrs', float, plottable=True)
        self.stats.register('momenta', float, plottable=True)
        
        self.reset(seed)  # sets random state for the beginning of the episode (in case it's changed during __init__)
        pass    
    
    
class LDS(DynamicalSystem):
    def __init__(self,
                 state_dim: int,
                 control_dim: int,
                 disturbance_type: str,
                 cost_fn: Union[str, Callable[[jnp.ndarray], float]],  # must be either 'quad', 'l1', or a function mapping states to costs
                 A: jnp.ndarray = None,
                 B: jnp.ndarray = None,
                 R: jnp.ndarray = None,
                 seed: int = None,
                 ):
        """
        Linear dynamical system of the form        
                `x_{t+1} = A @ x_t + B @ u_t + w_t,`
        where the `x_t` is state, `u_t` is control, and `w_t` is disturbance.
        
        Returns costs given by `f(x_t, u_t) = cost_fn(x_{t+1}) + u_t^T @ R @ u_t`

        Parameters
        ----------
        state_dim : int
            dimension of state
        control_dim : int
            dimensinon of control
        disturbance_type : str
            Must be one of `['none', 'constant', 'gaussian', 'sinusoidal', 'linear']`
        cost_fn : Union[str, Callable[[jnp.ndarray], float]]
            Must be one of `['quad', 'l1']` or a suitable function
        A : jnp.ndarray, optional
            state dynamics matrix, must be of shape `(state_dim, state_dim)`, by default random
        B : jnp.ndarray, optional
            control dynamics matrix, must be of shape `(state_dim, control_dim)`, by default random
        R : jnp.ndarray, optional
            matrix for quadratic control costs (i.e. `cost = cost_fn(x) + u^T @ R @ u`), by default
        """
        set_seed(seed)  # for reproducibility
        
        self.OBSERVABLE = True
        
        # figure out dynamics
        self.state_dim = state_dim
        self.control_dim = control_dim
        if A is None:
            done = False
            while not done:
                A = jax.random.normal(jkey(), (state_dim, state_dim))
                done = jnp.max(jnp.abs(jnp.linalg.eigvals(A))) < 1
        if B is None:
            B = sample(jkey(), (state_dim, control_dim), 'sphere')
        self.A, self.B = A, B
        assert self.A.shape == (state_dim, state_dim) and self.B.shape == (state_dim, control_dim)
        
        # figure out disturbances
        disturbance_fns = {'none': lambda t: 0.,
                           'constant': lambda t: 1.,
                           'gaussian': lambda t: np.random.randn() * 0.1,  # variance of 0.01
                        #    'linear': lambda t: float(t),  # this one is stupid
                           'sinusoidal': lambda t: np.sin(2 * np.pi * t / 100),  # period of 100 steps
                           'square wave': lambda t: np.ceil(t / 100) % 2  # period of 40 
                           }
        assert disturbance_type in disturbance_fns
        self.disturbance = disturbance_fns[disturbance_type]
        
        # figure out costs
        cost_fns = {'quad': lambda x: jnp.dot(x, x),
                    'hinge': lambda x: jnp.sum(jnp.abs(x))}
        if isinstance(cost_fn, str): 
            assert cost_fn in cost_fns
            cost_fn = cost_fns[cost_fn]
        self.R = R if R is not None else jnp.identity(control_dim)
        assert self.R.shape == (control_dim, control_dim)
        self.cost_fn = lambda x, u: cost_fn(x) + u.T @ self.R @ u
        
        # figure out stats to keep track of
        self.t = 0
        self.stats = Stats()
        self.stats.register('xs', jnp.ndarray, plottable=True)
        self.stats.register('us', jnp.ndarray, plottable=True)
        # self.stats.register('ws', float, plottable=True)
        self.stats.register('fs', float, plottable=True)
        
        # figure out init
        self.initial_state = jax.random.normal(jkey(), (state_dim,))
        self.reset(seed)  # sets random state for the beginning of the episode (in case it's changed during __init__)
        pass
    
    def reset(self, seed: int = None):
        super().reset(seed)
        self.state = self.initial_state.copy()
        return self

    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        assert control.shape == (self.control_dim,)
        
        # apply control
        state = self.A @ self.state + self.B @ control
        disturbance = self.disturbance(self.t)
        state = state + disturbance
        
        # observe costs
        cost = self.cost_fn(state, control)
        if hasattr(cost, 'item'): cost = cost.item()
        
        # update
        self.stats.update('xs', self.state, t=self.t)
        self.stats.update('us', control, t=self.t)
        # self.stats.update('ws', disturbance, t=self.t)
        self.stats.update('fs', cost, t=self.t)
        self.state = state
        self.t += 1
        
        return cost, state
        
        
class COCO(LDS):
    def __init__(self, 
                 index: int,
                 dim: int, 
                 disturbance_type: str, 
                 predict_differences: bool,
                 seed: int = None):
        
        # get problem
        suite_name = "bbob"
        suite = cocoex.Suite(suite_name, "", "")
        self.problem = suite[index]
        assert self.problem.dimension >= dim
        idxs = jnp.arange(self.problem.dimension)[:dim]
        
        A = jnp.identity(dim) if predict_differences else jnp.zeros((dim, dim))
        B = jnp.identity(dim) 
        R = jnp.identity(dim) # jnp.zeros((dim, dim))
        
        super().__init__(dim, dim, disturbance_type, None, A, B, R, seed)    
       
        full_x = jax.random.normal(jkey(), (self.problem.dimension,))  # handles the other coordinates
        def cost_fn(x: jnp.ndarray) -> float:
            sq_dist = jnp.sum(jnp.maximum(jnp.abs(x) - 5, 0) ** 2)  # how far past `[-5, 5]^dim` we are
            return self.problem(full_x.at[idxs].set(x)) + 0.1 * sq_dist    
        self.cost_fn = lambda x, u: cost_fn(x) + u.T @ self.R @ u
        
        # find optimal control
        _n = 10000
        _test = np.tile(full_x, _n).reshape(_n, -1)
        gt_xs = np.tile(np.linspace(-5, 5, _n), dim).reshape(dim, -1).T
        _test[:, idxs] = gt_xs
        gt_fs = [self.problem(_t) for _t in _test]
        xstar = gt_xs[np.argmin(gt_fs)]
        self.stats['xstar'] = xstar
        self.stats['gt_xs'] = gt_xs
        self.stats['gt_fs'] = gt_fs
        del _test
        pass

class MountainCar(DynamicalSystem):
    @abstractmethod
    def __init__(self, seed: int = None):
        # needs to set the following things
        set_seed(seed)  # for reproducibility
        self.stats: Stats = None
        self.OBSERVABLE: bool = None
        raise NotImplementedError()
    
    @abstractmethod
    def reset(self, seed: int = None):
        """
        to reset an episode, which should send state back to init
        """
        set_seed(seed)  # for reproducibility
        return self
    
    @abstractmethod
    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        given control, returns cost and an observation. The observation may be the true state, a function of the state, or simply `None`
        """
        raise NotImplementedError()