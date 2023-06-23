from abc import abstractmethod
import inspect
from typing import Tuple
from collections import deque
import random

import numpy as np
import jax.numpy as jnp

from models import MLP
from sysid import SysID
from utils import exponential_linspace_int, sample, set_seed, jkey, opnorm, dare_gain

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
               prev_state: jnp.ndarray,
               state: jnp.ndarray,
               control: jnp.ndarray,
               sysid: SysID) -> float:
        """
        Called with the previous state, the control that was applied, the resulting state, and the current
        estimate of system dynamics.
        Can be used to update the lifting mechanism, or can be a no-op.
        If updates, it can also return the loss.
        """
        return None  # by default a no-op
    
    
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

    def update(self,
               prev_state: jnp.ndarray,
               state: jnp.ndarray,
               control: jnp.ndarray,
               sysid: SysID):
        pass

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
                 batch_size: int = 10,
                 seed: int = None):
        
        super().__init__(hh, control_dim, state_dim, seed)
        
        self.hh = hh
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.scale = scale
        
        self.buffer = deque(maxlen=buffer_maxlen)
        self.batch_size = batch_size
        
        # to compute lifted states which hopefully respond linearly to the controls
        flat_dim = hh  # TODO could add control history as an input as well!
        self.lift_model = MLP(layer_dims=exponential_linspace_int(flat_dim, self.state_dim, depth), 
                              normalization = lambda dim: torch.nn.LayerNorm(dim),
                              use_bias=False, 
                              seed=seed).train().float()
        self.lift_opt = torch.optim.Adam(self.lift_model.parameters(), lr=lift_lr)
    
        # to estimate linear dynamics of lifted states
        self.A = torch.nn.Parameter(0.9 * torch.eye(self.state_dim, dtype=torch.float32))  # for stability purposes :)
        self.B = torch.nn.Parameter(0.5 * torch.randn((self.state_dim, self.control_dim), dtype=torch.float32))
        self.sysid_opt = torch.optim.Adam([self.A, self.B], lr=sysid_lr)
        
        self.cost_model = MLP(layer_dims=exponential_linspace_int(self.state_dim + self.control_dim, 1, depth),
                              seed=seed).train().float()
        self.cost_opt = torch.optim.Adam(self.cost_model.parameters(), lr=cost_lr)
        
        self.t = 1
        pass
    
    def forward(self, 
                cost_history: jnp.ndarray,
                control_history: jnp.ndarray) -> torch.Tensor:  # so that we don't need to return a jnp.ndarray
        
        # convert to pytorch tensors rq
        cost_history, control_history = map(lambda j_arr: torch.from_numpy(np.array(j_arr)), [cost_history, control_history])
        state = self.lift_model(cost_history.unsqueeze(0)).squeeze()
        return state
    
    def map(self,
            cost_history: jnp.ndarray,
            control_history: jnp.ndarray) -> jnp.ndarray:
        """
        maps histories to lifted states. it's in pytorch rn, but that can change
        """
        assert cost_history.shape == (self.hh,)
        assert control_history.shape == (self.hh, self.control_dim)
        
        state = self.forward(cost_history, control_history)
        state = jnp.array(state.data.numpy())  # convert back to jnp  # TODO remove this eventually
        
        return state
    
    def perturb_control(self,
                        state: jnp.ndarray,
                        control: jnp.ndarray = None):
        assert state.shape == (self.state_dim,)
        eps = sample(jkey(), (self.control_dim,))  # random direction
        control = self.scale * eps
        self.t += 1
        return control
    
    def sysid(self):
        return jnp.array(self.A.data.numpy()), jnp.array(self.B.data.numpy())

    # def learn(self):
    #     sample = random.sample(self.buffer, self.batch_size)
        
    #     losses = 0.
    #     for prev_histories, histories, control in sample:
    #         # convert
    #         if isinstance(control, jnp.ndarray):
    #             control = torch.from_numpy(np.array(control)).reshape(self.control_dim) 
    #         elif not isinstance(control, torch.Tensor): 
    #             control = torch.tensor(control).reshape(self.control_dim)
    #         if isinstance(prev_histories, jnp.ndarray):
    #             prev_histories = [torch.from_numpy(np.array(arr)) for arr in prev_histories]
    #             histories = [torch.from_numpy(np.array(arr)) for arr in histories]
            
    #         # compute disturbance
    #         prev_state = self.forward(*prev_histories)
    #         state = self.forward(*histories)
    #         pred = self.A @ prev_state + self.B @ control
    #         diff = state - pred
            
    #         # update
    #         self.lift_opt.zero_grad()
    #         self.sysid_opt.zero_grad()
            
    #         # compute loss 
    #         LAMBDA_NORM, LAMBDA_STABILITY = 1e-5, 1e-6
    #         norms = torch.linalg.norm(state) + torch.linalg.norm(prev_state)
    #         norm = norms + 1 / (norms + 1e-8)
    #         stability = opnorm(self.A - self.B @ dare_gain(self.A, self.B, torch.eye(self.state_dim), torch.eye(self.control_dim))) if LAMBDA_STABILITY > 0 else 0.
    #         loss = torch.mean(diff ** 2) + LAMBDA_NORM * norm + LAMBDA_STABILITY * stability
    #         loss.backward()
    #         self.lift_opt.step()
    #         self.sysid_opt.step()
            
    #         # self.cost_opt.zero_grad()
    #         # fhat = self.cost_model(torch.cat((prev_state.detach(), control), dim=-1).reshape(-1, self.state_dim + self.control_dim))  # predict cost from state and control we played from that state
    #         # f = histories[0][-1]
    #         # loss = torch.nn.functional.mse_loss(fhat.squeeze(), torch.from_numpy(np.array(f)).squeeze())
    #         # loss.backward()
    #         # self.cost_opt.step()
            
    #         losses += loss.item()
            
    #     losses /= self.batch_size
    #     return losses

    # def update(self,
    #            prev_histories: Tuple[jnp.ndarray, jnp.ndarray],
    #            histories: Tuple[jnp.ndarray, jnp.ndarray],
    #            control: jnp.ndarray,
    #            sysid: SysID) -> float:
        
    #     transition = (prev_histories, histories, control)
    #     self.buffer.append(transition)
        
    #     if len(self.buffer) >= self.batch_size:
    #         loss = self.learn()
    #     else:
    #         loss = None
    #     return loss
     