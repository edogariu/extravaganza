import logging
from abc import abstractmethod
from collections import defaultdict
from typing import List, Callable, Tuple
from copy import deepcopy

import numpy as np
import jax.numpy as jnp
import torch
import pykoopman as pk

from extravaganza.controllers import Controller
from extravaganza.observables import Trajectory
from extravaganza.models import TorchMLP, TorchPC3
from extravaganza.stats import Stats
from extravaganza.utils import set_seed, jkey, sample, get_classname, least_squares, summarize_lds

KOOPMAN_METHODS = ['polynomial', 'rbf', 'fourier']
MAX_OPNORM = None  # set this to `None` to have unconstrained opnorm of `A`
LOSS_WEIGHTS = {
    'jac': 0,
    'l2 linearization': 1,
    'l1 linearization': 0,
    'reconstruction': 0,
    'cpc': 0,
    'guess the control': 0,
    'simplification': 1,
    'residual centeredness': 0,
    'centeredness': 0
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

        set_seed(seed)
        
        self.obs_dim: int = obs_dim
        self.control_dim: int = control_dim
        self.state_dim: int = state_dim
        self.initial_control = jnp.zeros(self.control_dim) if initial_control is None else initial_control
        assert self.initial_control.shape == (self.control_dim,)
        
        if isinstance(exploration_scales, float): exploration_scales = [exploration_scales for _ in range(self.control_dim)]
        exploration_scales = jnp.array(np.array(exploration_scales))
        assert exploration_scales.shape == (self.control_dim,)
        self.exploration_scales = exploration_scales
        
        if exploration_bounds is not None:
            if isinstance(exploration_bounds[0], float): 
                assert len(exploration_bounds) == 2
                exploration_bounds = [exploration_bounds for _ in range(self.control_dim)]
            exploration_bounds = jnp.array(np.array(exploration_bounds))
            assert exploration_bounds.shape == (self.control_dim, 2)
        self.exploration_bounds = exploration_bounds
        
        self.max_traj_len: int = max_traj_len
        self.trajs: List[Trajectory] = [Trajectory()]
        
        for d in [obs_dim, control_dim, state_dim, max_traj_len]: assert d > 0
        
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
        if len(self.trajs[-1]) > 0: self.trajs.append(Trajectory())
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
                 AB_method: str = 'learned',
                 depth: int = 4,  # depth of NN
                 sigma: float = 0.,
                 determinstic_encoder: bool = False,
                 num_epochs: int = 20,
                 lifter_lr: float = 0.001,
                 
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
            
            mid_dim = 10 * (self.obs_dim + self.state_dim)
            layer_dims = [self.obs_dim, *[mid_dim for _ in range(depth - 1)], self.state_dim]
            enc = TorchMLP(layer_dims,
                           activation=torch.nn.ReLU,
                        #    normalization=torch.nn.LayerNorm,
                           drop_last_activation=True,
                           use_bias=True,
                           seed=seed)
            if LOSS_WEIGHTS['reconstruction'] > 0:
                layer_dims.reverse()
                dec = TorchMLP(layer_dims,
                            activation=torch.nn.ReLU,
                            #   normalization=torch.nn.LayerNorm,
                            drop_last_activation=True,
                            use_bias=True,
                            seed=seed)
            else:
                dec = None
            self.lifter = TorchPC3(enc, self.obs_dim, self.control_dim, self.state_dim, 
                                   AB_method=AB_method, decoder=dec, sigma=sigma, determinstic_encoder=determinstic_encoder, 
                                   do_cpc=LOSS_WEIGHTS['cpc'] > 0, do_jac=LOSS_WEIGHTS['jac'] > 0).float()
            self.lifter_opt = torch.optim.Adam(self.lifter.parameters(), lr=lifter_lr)

        else: raise NotImplementedError(method)
        
        self.trained = False
        pass
    
    def dynamics(self, z, u): return (self.A @ z.unsqueeze(-1) + self.B @ u.unsqueeze(-1)).squeeze(-1)
    
    def end_exploration(self, wordy: bool=True):
        logging.info('({}): ending sysid phase at step {}'.format(get_classname(self), self.t))        
        states, controls, costs = self.concatenate_trajectories()
        assert states.shape[0] == controls.shape[0] and states.shape[0] == costs.shape[0]
        
        # find the index of the last datapoint in each trajectory. we will ignore loss terms referencing x_t and x_{t+1} for all t in ignore_idxs,
        # since a reset happened between those two points
        traj_lens = [len(traj) for traj in self.trajs]
        ignore_idxs = np.cumsum(traj_lens) - 1  # -1 to capture last idxs of old trajs, not first idxs of new trajs
        mask = np.ones(states.shape[0], dtype=bool)
        mask[ignore_idxs] = False
        mask = mask[:-1]
        
        if self.method == 'identity':
            self.A, self.B = least_squares(states, controls, mask=mask, max_opnorm=MAX_OPNORM)
        
        elif self.method in KOOPMAN_METHODS:
            self.model.fit(x=np.array(states), u=np.array(controls))
            self.A = jnp.array(self.model.A.reshape(self.state_dim, self.state_dim))
            self.B = jnp.array(self.model.B.reshape(self.state_dim, self.control_dim))
            
        elif self.method == 'nn':  
            for k in LOSS_WEIGHTS.keys(): self.stats.register(k, obj_class=float)
                      
            # ----------------------------
            # DATASET
            # ----------------------------
            
            x, u, f = map(lambda arr: torch.tensor(np.array(arr)), [states, controls, costs])  # convert to tensors
            
            # normalize observations to be centered with unit variance. DONT TOUCH CONTROLS
            mean, std = torch.mean(x, dim=0), torch.std(x, dim=0)
            self.normalize = lambda t: (t - mean) / std
            x = self.normalize(x)
            
            # normalize costs to be in [0, 1], assuming they were nonnegative to begin with. DONT TOUCH CONTROLS
            fmin, fmax = torch.min(f), torch.max(f)
            # def unnormalize(t):
            #     gt_cost = (torch.linalg.norm(t, dim=-1) ** 2 * (fmax - fmin) + fmin).unsqueeze(-1)
            #     t = t * torch.sqrt(gt_cost) / torch.linalg.norm(t, dim=-1).unsqueeze(-1)
            #     assert torch.allclose(torch.linalg.norm(t, dim=-1) ** 2, gt_cost.squeeze(-1))
            #     return t
            self.unnormalize = lambda t: t * torch.sqrt(fmax)
            f = f / fmax
            
            mask = torch.tensor(np.array(mask))
            
            # sysid + train loop
            print_every = max(self.num_epochs // 10, 1)
            logging.info('training!')
            overall_losses = defaultdict(list)

            for i_epoch in range(self.num_epochs):
                
                self.lifter_opt.zero_grad()
                
                (z, zhat), (A, B), losses = self.lifter.get_embs_and_losses(x, u, mask)   
                zprev, zgt = z[:-1], z[1:]             
                
                # linearization error
                disturbances = (zgt - zhat)[mask]
                if LOSS_WEIGHTS['l2 linearization'] > 0: losses['l2 linearization'] = torch.nn.functional.mse_loss(disturbances, torch.zeros_like(disturbances))
                if LOSS_WEIGHTS['l1 linearization'] > 0: losses['l1 linearization'] = torch.nn.functional.l1_loss(disturbances, torch.zeros_like(disturbances))
                
                # how well we can reproduce the controls we used --   znext = A @ z + B @ u  ->  B^-1 @ (znext - A @ z) = u
                if LOSS_WEIGHTS['guess the control'] > 0: 
                    uhat = (torch.linalg.pinv(B) @ (zgt.unsqueeze(-1) - (A @ zprev.unsqueeze(-1)))).squeeze(-1) 
                    losses['guess the control'] = torch.nn.functional.mse_loss(uhat[mask], u[:-1][mask])
                
                # how well the squared norm represents cost
                if LOSS_WEIGHTS['simplification'] > 0: losses['simplification'] = torch.nn.functional.mse_loss(torch.linalg.norm(z, dim=-1) ** 2, f)
                
                # centeredness
                if LOSS_WEIGHTS['residual centeredness'] > 0: losses['residual centeredness'] = torch.linalg.norm(disturbances.mean(dim=0)) ** 2  # sq norm of mean residual
                if LOSS_WEIGHTS['centeredness'] > 0: losses['centeredness'] = torch.linalg.norm(z.mean(dim=0)) ** 2  # sq norm of mean embedding
                
                loss = 0.
                for k, v in losses.items(): 
                    l = LOSS_WEIGHTS[k] * v
                    loss += l
                    overall_losses[k].append(l.item())
                loss.backward()
                self.lifter_opt.step()
                
                if i_epoch % print_every == 0 or i_epoch == self.num_epochs - 1:
                    logging.info('mean loss for epochs {} - {}:'.format(i_epoch - print_every, i_epoch))
                    for k, v in overall_losses.items(): logging.info('\t\t{}: {}'.format(k, np.mean(v[-print_every:])))
                
            with torch.no_grad():
                z = jnp.array(self.unnormalize(self.lifter.encode(self.normalize(torch.tensor(np.array(states))))).detach().data.numpy())
                self.A, self.B = least_squares(z, controls, max_opnorm=MAX_OPNORM)
        
        if hasattr(self.lifter, 'A'):
            self.lifter.A.detach_(); self.lifter.B.detach_()
            self.lifter.B *= fmax ** 0.5
            print(fmax)
            
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
                x = torch.tensor(np.array(obs.reshape(1, -1)), dtype=torch.float32)
                x = self.normalize(x)
                z = self.lifter.encode(x)
                z = self.unnormalize(z)
                state = jnp.array(z.squeeze(0).data.numpy())
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
        # lifter.stats = self.stats
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
    