import logging
from abc import abstractmethod
import inspect
from typing import Tuple
from collections import deque

import numpy as np
import jax.numpy as jnp

from extravaganza.models import MLP
from extravaganza.sysid import SysID
from extravaganza.utils import exponential_linspace_int, sample, set_seed, jkey, opnorm, dare_gain, get_classname

class Lifter:
    """
    Map the past `hh` costs and controls to lifted "states".
    The given cost and control histories should be rightmost recent.
    """
    @abstractmethod
    def __init__(self,
                 hh: int,
                 control_dim: int,
                 state_dim: int,
                 seed: int = None):
        self.hh = hh
        self.control_dim = control_dim
        self.state_dim = state_dim
        set_seed(seed)
        pass
    
    @abstractmethod
    def map(self,
            cost_history: jnp.ndarray,
            control_history: jnp.ndarray) -> jnp.ndarray:
        """
        Maps a history of costs and controls to a "state" (which may just be the cost history or may be a 
        lifted state or anything).
        """
        raise NotImplementedError('{} not implemented'.format(inspect.stack()[0][3]))

    @abstractmethod
    def update(self,
               prev_histories: Tuple[jnp.ndarray, jnp.ndarray],
               histories: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
        """
        Called with the previous state and the resulting state, given as `(cost, control)` history tuples.
        Can be used to update the lifting mechanism, or can be a no-op.
        If updates, it can also return the loss for stat logging.
        """
        return 0.  # by default a no-op
    
    
# ---------------------------------------------------------------------------------------------------

class NoLift(Lifter):
    """
    lift to state that is simply history of costs
    """
    def __init__(self,
                 hh: int,
                 control_dim: int,
                 seed: int = None):
        super().__init__(hh, control_dim, hh, seed)
        pass
    
    def map(self,
            cost_history: jnp.ndarray,
            control_history: jnp.ndarray) -> jnp.ndarray:
        """
        maps histories to lifted states. it's in pytorch rn, but that can change
        """
        assert cost_history.shape == (self.hh,)
        assert control_history.shape == (self.hh, self.control_dim)
        
        return cost_history
    
# ---------------------------------------------------------------------------------------------------

import torch
"""
at the moment DL is in pytorch, once its finalized i can convert to jax for efficiency :)
"""

class RandomLift(Lifter):
    """
    Uses a random init NN to transform the cost+control histories into a system that hopefully has linear dynamics.
    """
    def __init__(self,
                 hh: int,
                 control_dim: int,
                 state_dim: int,
                 depth: int,
                 seed: int = None):
        super().__init__(hh, control_dim, state_dim, seed)
        
        self.hh = hh
        self.control_dim = control_dim
        self.state_dim = state_dim
        
        # to compute lifted states which hopefully respond linearly to the controls
        flat_dim = hh  # TODO could add control history as an input as well!
        self.lift_model = MLP(layer_dims = exponential_linspace_int(flat_dim, self.state_dim, depth), 
                              normalization = lambda dim: torch.nn.LayerNorm(dim),
                              use_bias = False, 
                              seed = seed).train().float()
        pass
    
    
    def map(self,
            cost_history: jnp.ndarray,
            control_history: jnp.ndarray) -> jnp.ndarray:
        """
        maps histories to lifted states. it's in pytorch rn, but that can change
        """
        assert cost_history.shape == (self.hh,)
        assert control_history.shape == (self.hh, self.control_dim)
        
        # convert to pytorch tensors and back rq
        with torch.no_grad():
            cost_history, control_history = map(lambda j_arr: torch.from_numpy(np.array(j_arr)), [cost_history, control_history])
            state = self.lift_model(cost_history.unsqueeze(0)).squeeze()
        state = jnp.array(state.cpu().data)
        
        return state

# ---------------------------------------------------------------------------------------------------

class LearnedLift(Lifter, SysID):
    def __init__(self,
                 hh: int,
                 control_dim: int,
                 state_dim: int,
                 depth: int,
                 scale: float = 0.1,
                 lift_lr: float = 0.001,
                 sysid_lr: float = 0.001,
                 cost_lr: float = 0.001,
                 buffer_maxlen: int = int(1e9),
                 batch_size: int = 64,
                 num_epochs: int = 20,  # number of epochs over the buffer to use when querying `sysid()` or `dynamics()` for first time
                 seed: int = None):
        
        set_seed(seed)
        super().__init__(hh, control_dim, state_dim, seed)
        
        self.hh = hh
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.scale = scale
        
        self.buffer = deque(maxlen=buffer_maxlen)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # to compute lifted states which hopefully respond linearly to the controls
        flat_dim = hh + control_dim * hh  # TODO could add control history as an input as well!
        self.lift_model = MLP(layer_dims=exponential_linspace_int(flat_dim, self.state_dim, depth), 
                              normalization = lambda dim: torch.nn.LayerNorm(dim),
                              use_bias=False, 
                              seed=seed).train().float()
        self.lift_opt = torch.optim.Adam(self.lift_model.parameters(), lr=lift_lr)
    
        # to estimate linear dynamics of lifted states
        self.A = torch.nn.Parameter(0.99 * torch.eye(self.state_dim, dtype=torch.float32))  # for stability purposes :)
        self.B = torch.nn.Parameter(torch.zeros((self.state_dim, self.control_dim), dtype=torch.float32))
        self.sysid_opt = torch.optim.Adam([self.A, self.B], lr=sysid_lr)
        
        # to learn "inverse" of lifing function
        self.cost_model = MLP(layer_dims=exponential_linspace_int(self.state_dim + self.control_dim, 1, depth),
                              seed=seed).train().float()
        self.cost_opt = torch.optim.Adam(self.cost_model.parameters(), lr=cost_lr)
        
        self.t = 1
        self.trained = False
        pass
    
    def forward(self, 
                cost_history: torch.Tensor,
                control_history: torch.Tensor) -> torch.Tensor:  # so that we don't need to return a jnp.ndarray
        inp = torch.cat((cost_history.reshape(-1, self.hh), control_history.reshape(-1, self.control_dim * self.hh)), dim=-1)
        state = self.lift_model(inp)
        return state
    
    def map(self,
            cost_history: jnp.ndarray,
            control_history: jnp.ndarray) -> jnp.ndarray:
        """
        maps histories to lifted states. it's in pytorch rn, but that can change
        """
        assert cost_history.shape == (self.hh,)
        assert control_history.shape == (self.hh, self.control_dim)
        
        # convert to pytorch tensors and back rq  # TODO remove this eventually, making everything in jax
        with torch.no_grad():
            cost_history, control_history = map(lambda j_arr: torch.from_numpy(np.array(j_arr)).unsqueeze(0), [cost_history, control_history])
            state = jnp.array(self.forward(cost_history, control_history).squeeze(0).data.numpy())
        
        return state
    
    def perturb_control(self,
                        state: jnp.ndarray,
                        control: jnp.ndarray = None):
        assert state.shape == (self.state_dim,)
        eps = sample(jkey(), (self.control_dim,))  # random direction
        control = self.scale * eps
        self.t += 1
        return control
    
    def train(self):
        logging.info('({}): training!'.format(get_classname(self)))
            
        # prepare dataloader
        from torch.utils.data import DataLoader, TensorDataset
        controls = []
        prev_cost_history = []
        prev_control_history = []
        cost_history = []
        control_history = []
        for prev_histories, histories in self.buffer:  # append em all
            lists = [controls, prev_cost_history, prev_control_history, cost_history, control_history]
            vals = [histories[1][-1], *prev_histories, *histories]
            for l, v in zip(lists, vals): l.append(torch.from_numpy(np.array(v)))
        dataset = TensorDataset(*map(lambda l: torch.stack(l, dim=0), 
                                     [prev_cost_history, prev_control_history, cost_history, control_history, controls]))
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        losses = []
        for t in range(self.num_epochs):
            for prev_cost_history, prev_control_history, cost_history, control_history, controls in dl:
                
                # compute disturbance
                prev_state = self.forward(prev_cost_history, prev_control_history)
                state = self.forward(cost_history, control_history)
                pred = self.A.expand(self.batch_size, self.state_dim, self.state_dim) @ prev_state.unsqueeze(-1) + \
                       self.B.expand(self.batch_size, self.state_dim, self.control_dim) @ controls.unsqueeze(-1)
                diff = state - pred.squeeze(-1)
                
                # update
                self.lift_opt.zero_grad()
                self.sysid_opt.zero_grad()
                
                # compute loss
                LAMBDA_STATE_NORM, LAMBDA_STABILITY, LAMBDA_B_NORM = 1e-6, 0, 1e-5
                norm = torch.norm(state)
                state_norm = (1 / (norm + 1e-8)) + norm if LAMBDA_STATE_NORM > 0 else 0.
                stability = opnorm(self.A - self.B @ dare_gain(self.A, self.B, torch.eye(self.state_dim), torch.eye(self.control_dim))) / opnorm(self.A) if LAMBDA_STABILITY > 0 else 0.
                B_norm = 1 / (torch.norm(self.B) + 1e-8) if LAMBDA_B_NORM > 0 else 0.
                loss = torch.mean(torch.abs(diff)) + LAMBDA_STATE_NORM * state_norm + LAMBDA_STABILITY * stability + LAMBDA_B_NORM * B_norm
                loss.backward()
                self.lift_opt.step()
                self.sysid_opt.step()
                
                # self.cost_opt.zero_grad()
                # fhat = self.cost_model(torch.cat((prev_state.detach(), control), dim=-1).reshape(-1, self.state_dim + self.control_dim))  # predict cost from state and control we played from that state
                # f = histories[0][-1]
                # loss = torch.nn.functional.mse_loss(fhat.squeeze(), torch.from_numpy(np.array(f)).squeeze())
                # loss.backward()
                # self.cost_opt.step()
                
                losses.append(loss.item())
                
            print_every = 25
            if t % print_every == 0 or t == self.num_epochs - 1: 
                logging.info('({}) \tmean loss for past {} epochs was {}'.format(get_classname(self), print_every, np.mean(losses[-print_every:])))
            
        self.trained = True
        return losses
    
    def sysid(self):
        if not self.trained:
            self.losses = self.train()
            A, B = jnp.array(self.A.data.numpy()), jnp.array(self.B.data.numpy())
            logging.info('({}) ||A||_op = {}     ||B||_F {}'.format(get_classname(self), opnorm(A), jnp.linalg.norm(B, 'fro')))
            return A, B
        
        A, B = jnp.array(self.A.data.numpy()), jnp.array(self.B.data.numpy())
        return A, B
    
    def update(self, 
               prev_histories: Tuple[jnp.ndarray, jnp.ndarray], 
               histories: Tuple[jnp.ndarray, jnp.ndarray]) -> float:
        self.buffer.append((prev_histories, histories))  # add the transition
        return 0.
