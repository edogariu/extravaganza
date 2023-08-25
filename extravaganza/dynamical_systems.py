import logging
from abc import abstractmethod
from typing import Callable, Tuple, Union, Iterator
from collections import defaultdict
from dataclasses import dataclass

from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
import cocoex

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import jax
import jax.numpy as jnp

import gymnasium as gym

from extravaganza.controllers import PID, PPO
from extravaganza.models import TorchMLP, TorchCNN
from extravaganza.stats import Stats
from extravaganza.utils import device, set_seed, jkey, get_classname, ContinuousCartPoleEnv, random_lds
                
class DynamicalSystem:
    reset_hook: Callable = lambda self: None
    
    @abstractmethod
    def __init__(self, seed: int = None, stats: Stats = None):
        # needs to set the following things
        set_seed(seed)  # for reproducibility
        self.stats = stats
        self.OBSERVABLE: bool = None
        raise NotImplementedError()
    
    @abstractmethod
    def reset(self, seed: int = None):
        """
        to reset an episode, which should send state back to init
        """
        set_seed(seed)  # for reproducibility
        self.reset_hook()
        return self
    
    @abstractmethod
    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        given control, returns cost and an observation. The observation may be the true state, a function of the state, or simply `None`
        """
        raise NotImplementedError()
    
@dataclass
class NNTraining(DynamicalSystem):
    """
    neural network training is a dynamical system with learning rate and momentum as controls
    """
    # needs to set the following things
    stats: Stats
    model: torch.nn.Module   # this is basically the system state
    opt: torch.optim.Optimizer
    apply_control: Callable[[jnp.ndarray, DynamicalSystem], None]  # applies the control to the system (which might update the learning rate or momentum or something)
    train_dl: DataLoader
    val_dl: DataLoader
    dl: Iterator
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # loss(yhat, y)
    eval_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # loss(yhat, y)
    eval_every: int = 1
    repeat: int = 1
    t: int = 0
    episode_t: int = 1
        
    def reset(self, seed: int = None):
        """
        to reset an episode, which should send state back to init
        """
        set_seed(seed)  # for reproducibility
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                # try:
                #     torch.nn.init.xavier_uniform_(layer.weight)
                #     if hasattr(self.model, 'use_bias') and self.model.use_bias:
                #         torch.nn.init.xavier_uniform_(layer.bias)
                # except: pass
        self.model.train()  
        self.opt.__setstate__({'state': defaultdict(dict)})  
        
        if 'train losses' not in self.stats._stats: self.stats.register('train losses', obj_class=float)
        if 'val losses' not in self.stats._stats: self.stats.register('val losses', obj_class=float)
        if 'avg train losses since reset' not in self.stats._stats: self.stats.register('avg train losses since reset', obj_class=float)
        if 'avg val losses since reset' not in self.stats._stats: self.stats.register('avg val losses since reset', obj_class=float)
        if 'lrs' not in self.stats._stats: self.stats.register('lrs', obj_class=float)
        if 'momenta' not in self.stats._stats: self.stats.register('momenta', obj_class=float)
        self.episode_t = 1
        self.episode_trainlosses = []
        self.episode_vallosses = []
        
        self.reset_hook()
        return self
    
    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        # apply control
        self.apply_control(control, self)
        
        tloss = 0.
        for _ in range(self.repeat):
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
            tloss += train_loss.item()
        tloss /= self.repeat
        
        # update stats
        self.episode_trainlosses.append(tloss)
        self.stats.update('train losses', tloss, t=self.t)
        self.stats.update('avg train losses since reset', np.mean(self.episode_trainlosses), t=self.t)
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
                val_loss = val_loss.item()
                self.episode_vallosses.append(val_loss)
                self.stats.update('val losses', val_loss, t=self.t)
                self.stats.update('avg val losses since reset', np.mean(self.episode_vallosses), t=self.t)
            self.model.train()
            
        self.t += 1
        self.episode_t += 1

        return tloss, None

class LinearRegression(NNTraining):
    """
    GD over linear regression problem on one of the following datasets: `['california', 'diabetes', 'generated']`.
    Costs are measured in MSE loss values, and each interaction is one full GD update.
    """
    def __init__(self, 
                 make_optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer], 
                 apply_control: Callable[[jnp.ndarray, DynamicalSystem], None],
                 dataset: str = 'diabetes',
                 repeat: int = 1,
                 eval_every: int = None,
                 seed: int = None,
                 stats: Stats = None):
        
        set_seed(seed)  # for reproducibility
        self.OBSERVABLE = False

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
        train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=train_x.shape[0], shuffle=False)
        val_dl = DataLoader(TensorDataset(val_x, val_y), batch_size=val_x.shape[0], shuffle=False)
        dl = iter(train_dl)
        
        # model
        model = torch.nn.Linear(train_x.shape[1], 1).float()
        opt = make_optimizer(model)
        model.train().to(device)
        
        # losses
        loss_fn = eval_fn = lambda yhat, y: torch.nn.functional.mse_loss(yhat, y)
        
        # stats
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
            
        super().__init__(stats=stats, model=model, opt=opt, apply_control=apply_control, 
                                         train_dl=train_dl, val_dl=val_dl, dl=dl,
                                         loss_fn=loss_fn, eval_fn=eval_fn, 
                                         repeat=repeat, eval_every=eval_every)
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
                 repeat: int = 1,
                 eval_every: int = None,
                 seed: int = None,
                 stats: Stats = None):
        
        set_seed(seed)  # for reproducibility
        self.OBSERVABLE = False

        # data
        train_dl = DataLoader(
            torchvision.datasets.MNIST('.data', train=True, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, drop_last=True)
        val_dl = DataLoader(
            torchvision.datasets.MNIST('./data', train=False, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=False, drop_last=True)
        dl = iter(train_dl)
        
        # model
        assert model_type in ['MLP', 'CNN']
        if model_type == 'MLP': model = TorchMLP(layer_dims=[int(28 * 28), 100, 100, 10]).float()
        elif model_type == 'CNN': model = TorchCNN(input_shape=(28, 28), output_dim=10)
        opt = make_optimizer(model)
        model.train().to(device)  
        
        # losses
        loss_fn = eval_fn = lambda yhat, y:  torch.nn.functional.nll_loss(yhat.softmax(dim=-1).log(), y)

        # stats
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(str(self.__class__).split('.')[-1]))
            stats = Stats()
        
        super().__init__(stats=stats, model=model, opt=opt, apply_control=apply_control, 
                                         train_dl=train_dl, val_dl=val_dl, dl=dl,
                                         loss_fn=loss_fn, eval_fn=eval_fn, 
                                         repeat=repeat, eval_every=eval_every)
        
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
                 stats: Stats = None,
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
        A, B = random_lds(state_dim=self.state_dim, control_dim=self.control_dim)  # random discrete, stable system
        self.A, self.B = jnp.array(A).reshape(state_dim, state_dim), jnp.array(B).reshape(state_dim, control_dim)
        
        # figure out disturbances
        disturbance_fns = {'none': lambda t: 0.,
                           'constant': lambda t: 1.,
                           'gaussian': lambda t: np.random.randn() * 0.1,  # variance of 0.01
                        #    'linear': lambda t: float(t),  # this one is stupid
                           'sinusoidal': lambda t: np.sin(2 * np.pi * t / 750),  # period of 500 750
                           'square wave': lambda t: np.ceil(t / 750) % 2  # period of 750 
                           }
        assert disturbance_type in disturbance_fns
        self.disturbance = disturbance_fns[disturbance_type]
        
        # figure out costs
        cost_fns = {'quad': lambda x: jnp.linalg.norm(x) ** 2,
                    'hinge': lambda x: jnp.sum(jnp.abs(x))}
        if isinstance(cost_fn, str): 
            assert cost_fn in cost_fns
            cost_fn = cost_fns[cost_fn]
        self.R = R if R is not None else jnp.identity(control_dim)
        assert self.R.shape == (control_dim, control_dim)
        self.cost_fn = lambda x, u: cost_fn(x) #+ u.T @ self.R @ u
        logging.debug('(LDS) for the LDS we are !!!NOT!!! reporting the costs with the `u.T @ R @ u` part')
        
        # figure out stats to keep track of
        self.t = 0
        self.episode_fs = []
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        self.stats.register('true disturbances', obj_class=float)
        
        # figure out init
        self.initial_state = jax.random.normal(jkey(), (state_dim,))
        logging.debug('({}): initial state is {}'.format(get_classname(self), self.initial_state))
        self.reset(seed)  # sets random state for the beginning of the episode (in case it's changed during __init__)
        pass
    
    def reset(self, seed: int = None):
        super().reset(seed)
        self.state = self.initial_state.copy()
        self.episode_fs = []
        self.reset_hook()
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
        self.episode_fs.append(cost)
        self.stats.update('true disturbances', disturbance, t=self.t)
        self.state = state
        self.t += 1
        
        return cost, state
        
        
class COCO(LDS):
    def __init__(self, 
                 index: int,
                 dim: int, 
                 disturbance_type: str, 
                 predict_differences: bool,
                 seed: int = None,
                 stats: Stats = None):
        
        # get problem
        suite_name = "bbob"
        suite = cocoex.Suite(suite_name, "", "")
        self.problem = suite[index]
        assert self.problem.dimension >= dim
        idxs = jnp.arange(self.problem.dimension)[:dim]
        
        A = jnp.identity(dim) if predict_differences else jnp.zeros((dim, dim))
        B = jnp.identity(dim)
        R = jnp.identity(dim) # jnp.zeros((dim, dim))
        
        super().__init__(dim, dim, disturbance_type, None, A, B, R, seed, stats)    
       
        full_x = jax.random.normal(jkey(), (self.problem.dimension,)).clip(-5, 5)  # handles the other coordinates
        lambda_dist = 1
        def cost_fn(x: jnp.ndarray) -> float:
            v = jnp.clip(x, -5, 5)
            cost = self.problem(full_x.at[idxs].set(v))
            sq_dist = ((x - v) ** 2).sum()  # how far past `[-5, 5]^dim` we are
            return cost + lambda_dist * sq_dist
        if predict_differences:
            self.cost_fn = lambda x, u: cost_fn(x) + u.T @ self.R @ u
        else:
            self.cost_fn = lambda x, u: cost_fn(x)
        
        # find optimal control
        n = 5000
        gt_xs = np.tile(np.linspace(-5, 5, n), dim).reshape(dim, -1).T
        gt_fs = [self.cost_fn(x, jnp.zeros(dim)) for x in gt_xs]
        xstar = gt_xs[np.argmin(gt_fs)]
        self.stats['xstar'] = xstar
        self.stats['gt_xs'] = gt_xs
        self.stats['gt_fs'] = gt_fs
        pass
    
    def reset(self, seed: int = None):
        set_seed(seed)
        state = jax.random.uniform(jkey(), (self.state_dim,), minval=-5, maxval=5)
        if hasattr(self, 'state'): logging.info('({}) state has been reset to {}. it was {}'.format(get_classname(self), state, self.state))
        self.state = state
        # self.state = self.initial_state.copy()
        self.episode_fs = []
        self.reset_hook()
        return self


class Gym(DynamicalSystem):
    def __init__(self, 
                 env_name: str,
                 use_reward_costs: bool = False,
                 send_done: bool = False,  # whether to send the string `'done'` as the state in `self.interact()` if we finished an episode
                 repeat: int = 1,
                 render: bool = False,
                 max_episode_len: int = None,
                 seed: int = None,
                 stats: Stats = None):

        set_seed(seed)  # for reproducibility
        self.OBSERVABLE = True
        self.repeat = repeat
        self.render = render
        self.use_reward_costs = use_reward_costs
        self.send_done = send_done
        self.max_episode_len = max_episode_len
        self.reset_seed = seed
        
        # env
        self.continuous_action_space = 'continuous' in env_name.lower()
        if env_name == 'CartPoleContinuous-v1':
            self.env = ContinuousCartPoleEnv()
            if render: logging.warning('({}) i havent yet set up rendering for continuous cartpole :)'.format(get_classname(self)))
        else: self.env = gym.make(env_name, render_mode='human' if render else None)
        self.env_name = env_name
        self.state_dim = self.env.observation_space.shape[0]
        self.control_dim = self.env.action_space.shape[0] if self.continuous_action_space else 1
        
        if env_name == 'MountainCarContinuous-v0': self.cost_fn = lambda state: abs(0.45 - state[0])  # L1 distance from flag
        elif env_name in ['CartPole-v1', 'CartPoleContinuous-v1']: self.cost_fn = lambda state: state[2] ** 2  # MSE of pole angle
        elif env_name == 'Pendulum-v1': self.cost_fn = lambda state: (jnp.arctan2(state[1], state[0]) ** 2).item() + 0.1 * state[2].item() ** 2  # taken from https://gymnasium.farama.org/environments/classic_control/pendulum/
        else: raise NotImplementedError(env_name)
        self.initial_state, _ = self.env.reset()
        self.reset(self.reset_seed)
        
        # stats to keep track of
        self.t = 0
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        self.stats.register('fs', obj_class=float)
        self.stats.register('rewards', obj_class=float)
        pass
        
    def reset(self, seed: int = None):
        """
        to reset an episode, which should send state back to init
        """
        set_seed(seed)  # for reproducibility
        self.episode_reward = 0.
        self.episode_t = 0
        self.done = False
        self.state, _ = self.env.reset()
        self.reset_hook()
        return self
    

    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        given control, returns cost and an observation. The observation may be the true state, a function of the state, or simply `None`
        """
        if self.continuous_action_space: assert control.shape == (self.control_dim,), control.shape
        if self.max_episode_len is not None and self.episode_t > self.max_episode_len: self.done = True
        if self.done: self.reset(self.reset_seed)

        control = np.array(control)
        if self.env_name == 'CartPoleContinuous-v1': control = np.clip(control, -1, 1)
        
        cost = 0.
        for i in range(self.repeat):
            if self.max_episode_len is not None and self.episode_t > self.max_episode_len: self.done = True
            if self.done: break
            self.state, r, self.done, _, _ = self.env.step(control)# if self.continuous_action_space else control.item())
            self.episode_reward += r
            cost += self.cost_fn(self.state)
            self.episode_t += 1
        cost /= (i + 1)

        # update
        self.stats.update('fs', cost, t=self.t)
        self.stats.update('rewards', self.episode_reward, t=self.t)
        self.t += 1
        
        state = jnp.array(self.state)
        if self.use_reward_costs: cost = -self.episode_reward
        if self.done and self.send_done: state = 'done'
        
        return cost, state


class PIDGym(DynamicalSystem):
    def __init__(self, 
                 env_name: str,
                 apply_control: Callable[[jnp.ndarray, DynamicalSystem], None],
                 control_dim: int,
                 repeat: int = 1,
                 gym_repeat: int = 1,
                 max_episode_len: int = None,
                 Kp: jnp.ndarray = None, 
                 Ki: jnp.ndarray = None, 
                 Kd: jnp.ndarray = None, 
                 seed: int = None,
                 stats: Stats = None):
        
        set_seed(seed)
        self.OBSERVABLE = False
        self.apply_control = apply_control
        self.control_dim = control_dim
        self.repeat = repeat
        self.seed = seed
        
        # env
        gym_args = {
            'env_name': env_name,
            'use_reward_costs': False,
            'send_done': True,
            'repeat': gym_repeat,
            'render': False,
            'max_episode_len': max_episode_len,
            'seed': seed,
            'stats': stats,
        }
        self.gym = Gym(**gym_args)
        self.stats = self.gym.stats
        if env_name == 'MountainCarContinuous-v0': 
            setpoint = 0.45
            obs_mask = [1, 0]  # gets x position
        elif env_name in ['CartPole-v1', 'CartPoleContinuous-v1']: 
            setpoint = 0.
            obs_mask = [0, 0, 1, 0]  # gets pole angle
        else: raise NotImplementedError(env_name)
        
        # alg 
        pid_args = {
            'control_dim': self.gym.control_dim,
            'setpoint': setpoint,
            'obs_mask': obs_mask,
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
            'stats': self.stats
        }
        self.pid = PID(**pid_args)
        self.reset(seed)
        
        self.t = 0
        pass

    def reset(self, seed: int = None):
        """
        to reset an episode, which should send state back to init
        """
        # for reproducibility
        set_seed(seed)
        self.gym.reset(seed)
        self.state = None
    
        # reset controller history
        self.pid.reset(seed)
        
        self.reset_hook()
        return self

    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        given control, returns cost and an observation. The observation may be the true state, a function of the state, or simply `None`
        """
        assert control.shape == (self.control_dim,), control.shape
            
        # apply control
        self.apply_control(control, self)
        costs = 0.
        for _ in range(self.repeat):
            action = self.pid.get_control(None, self.state) if self.state is not None else self.gym.env.action_space.sample()
            
            if self.gym.continuous_action_space:
                action = jnp.clip(action, self.gym.env.action_space.low, self.gym.env.action_space.high) 
            else:   # if discrete cartpole
                action = 1 if action > 0 else 0

            cost, state = self.gym.interact(action)
            costs += cost
            if state == 'done':
                self.state = None
                self.pid.reset(self.seed)
                break
            else:
                self.state = state
        costs /= self.repeat
        self.t += 1
        
        return costs, None
        
        
class PPOGym(DynamicalSystem):
    def __init__(self, 
                 env_name: str,
                 apply_control: Callable[[jnp.ndarray, DynamicalSystem], None],
                 control_dim: int,
                 lr_actor: float = 0.0003,
                 lr_critic: float = 0.001,
                 eps_clip: float = 0.2,
                 gamma: float = 0.99,
                 repeat: int = 1,
                 gym_repeat: int = 1,
                 max_episode_len: int = None,
                 seed: int = None,
                 stats: Stats = None):
        
        set_seed(seed)  # for reproducibility
        
        self.OBSERVABLE = False
        self.apply_control = apply_control
        self.repeat = repeat
        self.control_dim = control_dim
        self.seed = seed
        
        # env
        gym_args = {
            'env_name': env_name,
            'use_reward_costs': True,
            'repeat': gym_repeat,
            'render': False,
            'max_episode_len': max_episode_len,
            'send_done': True,
            'seed': seed,
            'stats': stats,
        }
        self.gym = Gym(**gym_args)
        self.stats = self.gym.stats
        
        # alg
        ppo_args = {
            'state_dim': self.gym.env.observation_space.shape[0],
            'control_dim': self.gym.control_dim,
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'gamma': gamma,
            'eps_clip': eps_clip,
            'has_continuous_action_space': 'continuous' in env_name.lower(),
            'stats': self.stats,
        }
        self.ppo = PPO(**ppo_args)
        assert not self.ppo.continuous_action_space, 'i havent yet coded continuous PPO'
        
        self.reset(seed)
        self.t = 0        
        pass
    
    def reset(self, seed: int = None):
        """
        to reset an episode, which should send state back to init
        """
        # for reproducibility
        set_seed(seed)
        self.gym.reset(seed)
        self.state = None
        self.cost = None

        # reset controller
        self.ppo.reset(seed)
        
        self.reset_hook()
        return self
    
    def interact(self, control: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        given control, returns cost and an observation. The observation may be the true state, a function of the state, or simply `None`
        """
        assert control.shape == (self.control_dim,)
            
        # apply control
        self.apply_control(control, self)
        for _ in range(self.repeat):
            action = self.ppo.get_control(self.cost, self.state) if self.state is not None else self.gym.env.action_space.sample()
            action = jnp.array(action).reshape(self.control_dim)
            if self.state == 'done':
                self.state = None
                self.ppo.reset(self.seed)
                break
            self.cost, self.state = self.gym.interact(action)
        
        self.t += 1
        return self.cost, None
        