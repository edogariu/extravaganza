import logging
from abc import abstractmethod
from collections import defaultdict
from typing import List, Callable, Tuple
from copy import deepcopy

import numpy as np
import jax.numpy as jnp
import torch
import torch.utils.data as data
import pykoopman as pk

from extravaganza.controllers import Controller
from extravaganza.observables import Trajectory
from extravaganza.models import TorchMLP, TorchPC3
from extravaganza.stats import Stats
from extravaganza.utils import set_seed, jkey, sample, get_classname, least_squares, summarize_lds

KOOPMAN_METHODS = ['polynomial', 'rbf', 'fourier']
MAX_OPNORM = None  # set this to `None` to have unconstrained opnorm of `A`
LOSS_WEIGHTS = {
    'linearization': 1,
    'cpc': 0.5,
    'simplification': 0.1,
    'consistency': 0,
    'residual centeredness': 0,
    'centeredness': 1.
}

class SystemModel:
    def __init__(self,
                 obs_dim: int,
                 control_dim: int,
                 state_dim: int,
                 
                 max_traj_len: int,
                 exploration_scales: Tuple[float],
                
                 exploration_bounds: Tuple[Tuple[float]] = None,
                 initial_control: jnp.ndarray = None,
                 stats: Stats = None,
                 seed: int = None):

        for d in [obs_dim, control_dim, state_dim, max_traj_len, *exploration_scales]: assert d > 0
        set_seed(seed)
        
        self.obs_dim: int = obs_dim
        self.control_dim: int = control_dim
        self.state_dim: int = state_dim
        self.initial_control = jnp.zeros(self.control_dim) if initial_control is None else initial_control
        assert self.initial_control.shape == (self.control_dim,)
        
        assert len(exploration_scales) in [1, self.control_dim]
        if len(exploration_scales) == 1: exploration_scales = [exploration_scales[0] for _ in range(self.control_dim)]
        exploration_scales = jnp.array(np.array(exploration_scales))
        self.exploration_scales: Tuple[float] = exploration_scales
        
        if exploration_bounds is not None:
            assert len(exploration_bounds) in [1, self.control_dim]
            if len(exploration_bounds) == 1: exploration_bounds = [exploration_bounds[0] for _ in range(self.control_dim)]
            exploration_bounds = jnp.array(np.array(exploration_bounds))
        self.exploration_bounds: Tuple[Tuple[float]] = exploration_bounds
        
        self.max_traj_len: int = max_traj_len
        self.trajs: List[Trajectory] = [Trajectory()]
        
        # stats to keep track of
        self.trained = False
        self.t = 0
        self.stats = stats
        if self.stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            self.stats = Stats()
        pass
    
    def explore(self, cost: float, obs: jnp.ndarray):  # cost for entering into `obs`, where we are right now
        assert obs.shape == (self.obs_dim,), obs.shape

        # `control` is always the control played AFTER receiving `obs`
        self.trajs[-1].add_state(cost, obs)
        control = self.exploration_scales * sample(jkey(), shape=(self.control_dim,)) + self.initial_control
        if self.exploration_bounds is not None:
            for i in range(self.control_dim): control = control.at[i].set(jnp.clip(control[i], *self.exploration_bounds[i]))
        self.trajs[-1].add_control(control)

        self.t += 1
        return control
    
    def end_trajectory(self):
        if len(self.trajs[-1].f) > 0: self.trajs.append(Trajectory())
        pass
    
    def concatenate_trajectories(self):
        x, u, f = deepcopy(self.trajs[0].x), deepcopy(self.trajs[0].u), deepcopy(self.trajs[0].f)
        for traj in self.trajs[1:]:
            x.extend(traj.x)
            u.extend(traj.u)
            f.extend(traj.f)
        x, u, f = map(lambda arr: jnp.stack(arr, axis=0), [x, u, f])
        return x, u, f
    
    @abstractmethod
    def end_exploration(self):
        """
        this is where the system model makes use of the exploration dataset to train and do whatever
        """
        self.trained = True
        pass
    
    @abstractmethod
    def get_state(self, 
                  obs: jnp.ndarray): 
        """
        should be identity operation if we are not lifting
        """
        pass
    

class Lifter(SystemModel):
    def __init__(self,
                 obs_dim: int,
                 control_dim: int,
                 state_dim: int,
                 
                 max_traj_len: int,
                 exploration_scales: Tuple[float],
                 
                 method: str,  # must be in ['identity', 'polynomial', rbf', 'fourier', 'nn']
                 depth: int = 4,  # depth of NN
                 sigma: float = 0.,
                 determinstic_encoder: bool = False,
                 num_epochs: int = 20,
                 batch_size: int = 64,
                 lifter_lr: float = 0.001,
                 sysid_lr: float = 0.001,
                 
                 initial_control: jnp.ndarray = None,
                 exploration_bounds: Tuple[Tuple[float]] = None,
                 stats: Stats = None,
                 seed: int = None):
        
        super().__init__(obs_dim=obs_dim, control_dim=control_dim, state_dim=state_dim, max_traj_len=max_traj_len, exploration_scales=exploration_scales, exploration_bounds=exploration_bounds, initial_control=initial_control, stats=stats, seed=seed)
        self.method = method
        
        # to compute lifted states which hopefully respond linearly to the controls
        if method == 'identity':
            pass
        
        elif method in KOOPMAN_METHODS:
            if method == 'fourier': assert (self.state_dim - self.obs_dim) % 2 == 0, '`state_dim - obs_dim` must be even (not {}) for Koopman fourier methods!'.format(self.state_dim - self.obs_dim)
            if method == 'polynomial': raise NotImplementedError('oopsie, for some reason polynomial observations isnt working yet')

            observables = {
                'polynomial': pk.observables.Polynomial(degree=3),
                'rbf': pk.observables.RadialBasisFunction(rbf_type='gauss', n_centers=self.state_dim - self.obs_dim, include_state=True),
                'fourier': pk.observables.RandomFourierFeatures(D=(self.state_dim - self.obs_dim) // 2, include_state=True),
            }
            self.model = pk.Koopman(observables=observables[method], regressor=pk.regression.EDMDc())

        elif method == 'nn':
            self.num_epochs = num_epochs
            self.batch_size = batch_size
            
            # layer_dims = exponential_linspace_int(self.obs_dim, self.state_dim, depth)
            mid_dim = self.obs_dim + self.state_dim
            layer_dims = [self.obs_dim, *[mid_dim for _ in range(depth - 1)], self.state_dim]
            mlp = TorchMLP(layer_dims,
                           activation=torch.nn.ReLU,
                           #   normalization=torch.nn.LayerNorm,
                           drop_last_activation=True,
                           use_bias=True,
                           seed=seed)
            self.A, self.B = torch.nn.Parameter(torch.eye(self.state_dim)), torch.nn.Parameter(torch.zeros((self.state_dim, self.control_dim)))
            dynamics_fn = lambda z, u: (self.A @ z.unsqueeze(-1) + self.B @ u.unsqueeze(-1)).squeeze(-1)
            self.lifter = TorchPC3(mlp, dynamics_fn, self.obs_dim, self.control_dim, self.state_dim, sigma=sigma, determinstic_encoder=determinstic_encoder)
            self.sysid_opt = torch.optim.SGD([self.A, self.B], lr=sysid_lr, weight_decay=1e-5)
            self.lifter_opt = torch.optim.SGD(self.lifter.parameters(), lr=lifter_lr, weight_decay=1e-5)

        else: raise NotImplementedError(method)
        
        self.trained = False
        pass
    
    def end_exploration(self, wordy: bool=True):
        logging.info('({}): ending sysid phase at step {}'.format(get_classname(self), self.t))        
        states, controls, costs = self.concatenate_trajectories()
        assert states.shape[0] == controls.shape[0] and states.shape[0] == costs.shape[0]
        
        if self.method == 'identity':
            self.A, self.B = least_squares(states, controls, max_opnorm=MAX_OPNORM)
        
        elif self.method in KOOPMAN_METHODS:
            self.model.fit(x=np.array(states), u=np.array(controls))
            self.A = jnp.array(self.model.A.reshape(self.state_dim, self.state_dim))
            self.B = jnp.array(self.model.B.reshape(self.state_dim, self.control_dim))
            
        elif self.method == 'nn':            
            # ----------------------------
            # DATASET
            # ----------------------------
            x, u, f = map(lambda arr: torch.tensor(np.array(arr)), [states, controls, costs])
            dl = data.DataLoader(data.TensorDataset(x[:-1], u[:-1], x[1:], f[1:]), batch_size=self.batch_size, shuffle=True, drop_last=True)
            
            for k in LOSS_WEIGHTS.keys(): self.stats.register(k, obj_class=float)
            
            # sysid + train loop
            print_every = 25
            logging.info('training!')
            losses = defaultdict(list)
            i_iter = 0
            for i_epoch in range(self.num_epochs):
                epoch_losses = defaultdict(float)
                for batch in dl:  # train lifter
                    self.sysid_opt.zero_grad()
                    self.lifter_opt.zero_grad()
                    
                    x_prev, u_prev, x, f = batch
                    (z_prev, z, zhat), batch_losses = self.lifter.get_embs_and_losses(x_prev, u_prev, x)  # this references self.A and self.B, and so we backprop to them!
                    
                    # do a lil sysid and make sure it looks right (note that this backprops through to the lifter as well!)
                    # NOTE perhaps we should do that globally, once an epoch, instead of every batch?
                    if LOSS_WEIGHTS['consistency'] > 0:
                        ret = torch.linalg.lstsq(torch.hstack((z_prev, u_prev)), z, rcond=-1).solution
                        A, B = ret[:self.state_dim].T, ret[self.state_dim:].T
                        _k = 'consistency'; batch_losses[_k] = torch.linalg.norm(A - self.A) + torch.linalg.norm(B - self.B)
                    else: _k = 'consistency'; batch_losses[_k] = torch.zeros(1)[0]
                    _k = 'simplification'; batch_losses[_k] = torch.abs(torch.linalg.norm(z, dim=-1) ** 2 - f).mean()
                    _k = 'residual centeredness'; batch_losses[_k] = ((z - zhat).mean(dim=0) ** 2).sum() # coordinatewise squared mean residual
                    _k = 'centeredness'; batch_losses[_k] = (z.mean(dim=0) ** 2).sum()
                    
                    loss = 0.
                    for k, v in batch_losses.items(): loss += LOSS_WEIGHTS[k] * v
                    loss.backward()
                    self.sysid_opt.step()
                    self.lifter_opt.step()
                    
                    for k, v in batch_losses.items(): 
                        v = v.item()
                        epoch_losses[k] += v / len(dl)
                        self.stats.update(k, value=v, t=i_iter)
                    i_iter += 1
                for k, v in epoch_losses.items(): losses[k].append(v)
                if i_epoch % print_every == 0 or i_epoch == self.num_epochs - 1:
                    logging.info('mean loss for epochs {} - {} was {}'.format(i_epoch - print_every, i_epoch, {k: np.mean(v[-print_every:]) for k, v in losses.items()}))
                
            with torch.no_grad():
                z = jnp.array(self.lifter.encode(torch.tensor(np.array(states))).detach().data.numpy())
                self.A, self.B = least_squares(z, controls, max_opnorm=MAX_OPNORM)
            
        if wordy: print(summarize_lds(self.A, self.B))
        self.trained = True
        return self.A, self.B
    
    def get_state(self, 
                  obs: jnp.ndarray):
        assert obs.shape == (self.obs_dim,), (obs.shape, self.obs_dim)
        if not self.trained: 
            logging.warning('({}): tried to use lifter during sysid phase. ENDING SYSID PHASE at step {}'.format(get_classname(self), self.t))
            self.end_exploration()
        
        if self.method == 'identity':
            state = obs
        elif self.method in KOOPMAN_METHODS:
            obs = np.array(obs)  # pykoopman works in numpy, not jax
            state = self.model.observables.transform(obs.reshape(1, self.obs_dim)).squeeze(0)
            state = jnp.array(state)
        elif self.method == 'nn':  # we do dl in torch, not jax
            with torch.no_grad():
                x = torch.tensor(np.array(obs.reshape(1, -1)))
                z = self.lifter.encode(x).squeeze(0).data.numpy()
                state = jnp.array(z)
        return state
    
# ---------------------------------------------------------------------------------------------------------------------
#         SYSID WRAPPER TO EXPLORE AND FIT A SYSTEMMODEL (either first or online & jointly with controller)
# ---------------------------------------------------------------------------------------------------------------------

class LiftedController(Controller):
    def __init__(self,
                 controller: Controller,
                 lifter: SystemModel):
        # check things make sense
        assert controller.control_dim == lifter.control_dim, 'inconsistent control dims!'
        assert controller.state_dim == lifter.state_dim, 'inconsistent state dims!'
        
        super().__init__()
        self.controller = controller
        self.lifter = lifter
        self.control_dim = controller.control_dim
        self.state_dim = lifter.obs_dim
        self.stats = Stats.concatenate((controller.stats, lifter.stats))
        controller.stats = self.stats
        lifter.stats = self.stats
        self.system_reset_hook = self.controller.system_reset_hook
        pass
    
    def get_control(self, cost: float, obs: jnp.ndarray) -> jnp.ndarray:
        assert obs.shape == (self.state_dim,), (obs.shape, self.state_dim)
        state = self.lifter.get_state(obs)
        return self.controller.get_control(cost, state)
    
    def update(self, state: jnp.ndarray, cost: float, control: jnp.ndarray, next_state: jnp.ndarray, next_cost: float):
        state, next_state = self.lifter.get_state(state), self.lifter.get_state(next_state)
        return self.controller.update(state, cost, control, next_state, next_cost)

class OfflineSysid(Controller):
    def __init__(self,
                 make_controller: Callable[[SystemModel], Controller],
                 sysid: SystemModel,
                 T0: int):        
        self.make_controller = make_controller
        self.sysid = sysid
        self.T0 = T0
        
        self.controller: Controller = None  # the moment we call get_control() and reach self.t == T0, `self.controller` will no longer be None
        
        super().__init__()
        self.t = 0
        self.state_dim = sysid.obs_dim
        self.control_dim = self.sysid.control_dim
        self.stats = self.sysid.stats
        pass

    def system_reset_hook(self):
        """
        This is a function that gets called every time the dynamical system we are controlling gets episodically reset.
        Here, we use it to alert our system model that the current trajectory has ended
        """
        if self.t < self.T0: self.sysid.end_trajectory()
        else: self.controller.system_reset_hook()
        pass
    
    def update(self, state: jnp.ndarray, cost: float, control: jnp.ndarray, next_state: jnp.ndarray, next_cost: float):
        if self.t < self.T0: return
        return self.controller.update(state, cost, control, next_state, next_cost)
    
    def get_control(self, cost: float, obs: jnp.ndarray) -> jnp.ndarray:
        assert obs.shape == (self.state_dim,), (obs.shape, self.state_dim)
        
        self.t += 1
        if self.t < self.T0: return self.sysid.explore(cost, obs)
        elif self.t == self.T0:
            logging.info('(SYSID WRAPPER) ending exploration at timestep {}'.format(self.t))
            self.sysid.end_exploration()
            
            # make the controller
            self.controller = self.make_controller(self.sysid)
            assert isinstance(self.controller, LiftedController), 'controller produced by `make_controller()` should be a LiftedController'
            assert self.controller.lifter is self.sysid, 'the controller\'s lifter should be the provided sysid, otherwise things might not work'
            assert self.controller.control_dim == self.control_dim, (self.controller.control_dim, self.control_dim)
            self.stats = self.controller.stats
            
        return self.controller.get_control(cost, obs)
    