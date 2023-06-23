from abc import abstractmethod
from typing import Tuple
from collections import deque

import jax.numpy as jnp

from lifters import Lifter
from sysid import SysID
from stats import Stats
from utils import rescale, d_rescale, inv_rescale, append, sample, set_seed, jkey, opnorm


class Controller:
    """
    abstract class for all controllers
    """
    @abstractmethod
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        """
        gets the next control as a function of the cost and arrived-in state from playing the previous control
        """
        pass


class LiftedBPC(Controller):
    def __init__(self,
                 h: int,
                 initial_u: jnp.ndarray,
                 rescalers,
                 initial_scales: Tuple[float, float, float],
                 lifter: Lifter,
                 sysid: SysID,
                 T0: int,
                 bounds = None,
                 method = 'REINFORCE',
                 K: jnp.ndarray = None,
                 step_every: int = 1,
                 use_sigmoid = True,
                 decay_scales = False,
                 K_every: int = None,
                 seed: int = None):

        set_seed(seed)  # for reproducibility
        
        # check things make sense
        assert lifter.state_dim == sysid.state_dim
        assert lifter.control_dim == sysid.control_dim and initial_u.shape[0] == lifter.control_dim
        self.control_dim = lifter.control_dim
        self.state_dim = lifter.state_dim
        assert method in ['FKM', 'REINFORCE']
        assert all(map(lambda i: i >= 0, initial_scales))
        if bounds is not None:
            bounds = jnp.array(bounds).reshape(2, -1)
            assert len(bounds[0]) == len(bounds[1]) and len(bounds[0]) == self.control_dim, 'improper bounds'
            assert all(map(lambda i: bounds[0, i] < bounds[1, i], range(self.control_dim))), 'improper bounds'
        if K is not None:
            assert K.shape == (self.control_dim, self.state_dim)
        assert step_every < h, 'need to update at least every `h` steps'
        
        # hyperparams
        self.h = h
        self.hh = lifter.hh
        self.lifter = lifter
        self.sysid = sysid
        self.T0 = T0
        self.method = method
        self.bounds = bounds
        self.decay_scales = decay_scales
        self.K_every = K_every
        self.initial_control = initial_u
        self.step_every = step_every

        # for rescaling u
        self.rescale_u = lambda u: rescale(u, self.bounds, use_sigmoid=use_sigmoid) if self.bounds is not None else u
        self.inv_rescale_u = lambda ru: inv_rescale(ru, self.bounds, use_sigmoid=use_sigmoid) if self.bounds is not None else ru
        self.d_rescale_u = lambda u: d_rescale(u, self.bounds, use_sigmoid=use_sigmoid) if self.bounds is not None else jnp.ones_like(u)        
        
        # controller params
        self.M = jnp.zeros((self.h, self.control_dim, self.state_dim))
        self.M0 = self.inv_rescale_u(initial_u)
        self.K = K if K is not None else jnp.zeros((self.control_dim, self.state_dim)) # jax.random.normal(self.jkey(), shape=(self.control_dim, self.state_dim)) / (self.control_dim * self.state_dim)
        self.M_scale, self.M0_scale, self.K_scale = initial_scales
        
        # histories are rightmost recent (increasing in time)
        self.prev_cost = 0.
        self.prev_control = jnp.zeros(self.control_dim)
        self.prev_state = jnp.zeros(self.state_dim)
        self.disturbance_history = jnp.zeros((2 * self.h, self.state_dim))  # past 2h disturbances, for controller
        self.cost_history = jnp.zeros(self.hh)  # for sysid/lifting
        self.control_history = jnp.zeros((self.hh, self.control_dim))  # for sysid/lifting
        self.t = 1

        # grad estimation stuff -- NOTE maybe `self.eps` should be divided by its variance?
        if self.method == 'FKM':
            self.eps_M = jnp.zeros((self.h, self.h, self.control_dim, self.state_dim))  # noise history of M perturbations
            self.eps_M0 = jnp.zeros((self.h, self.control_dim))  # noise history of M0 perturbations
            self.eps_K = jnp.zeros((self.h, self.control_dim, self.state_dim))  # noise history of K perturbations
            
            def grad_M(diff):
                return diff * jnp.sum(self.eps_M, axis=0) #* self.control_dim * self.state_dim * self.h
            def grad_M0(diff):
                return diff * jnp.sum(self.eps_M0, axis=0) #* self.control_dim * self.state_dim * self.h
            def grad_K(diff):
                return diff * jnp.sum(self.eps_K, axis=0) #* self.control_dim * self.state_dim * self.h
            
        elif self.method == 'REINFORCE':
            self.eps = jnp.zeros((self.h + 1, self.control_dim))  # noise history of u perturbations
            
            def grad_M(diff):
                val = sum([jnp.transpose(jnp.einsum('ij,k->ijk', self.disturbance_history[i: self.h + i], self.eps[i]), axes=(0, 2, 1)) for i in range(self.h)])
                return diff * val #* self.control_dim * self.state_dim * self.h
            def grad_M0(diff):
                return diff * self.eps[-1] #* self.control_dim * self.state_dim * self.h
            def grad_K(diff):
                val = self.eps[-1].reshape(self.control_dim, 1) @ self.prev_state.reshape(1, self.state_dim)
                return diff * val #* self.control_dim * self.state_dim * self.h
            
        self.grads = deque([(jnp.zeros_like(self.M), jnp.zeros_like(self.M0), jnp.zeros_like(self.K))], maxlen=self.h)
        self.grad_M = grad_M
        self.grad_M0 = grad_M0
        self.grad_K = grad_K
        
        self.M_update_rescaler = rescalers[0]()
        self.M0_update_rescaler = rescalers[1]()
        self.K_update_rescaler = rescalers[2]()
        
        # stats
        self.stats = Stats()
        self.stats.register('||A-BK||_op', float, plottable=True)
        self.stats.register('||A||_op', float, plottable=True)
        self.stats.register('||B||_F', float, plottable=True)
        self.stats.register('disturbances', float, plottable=True)
        self.stats.register('lifter losses', float, plottable=True)
        if self.control_dim == 1:
            self.stats.register('K @ state', float, plottable=True)
            self.stats.register('M \cdot w', float, plottable=True)
            self.stats.register('M0', float, plottable=True)
        pass

# ------------------------------------------------------------------------------------------------------------
    
    def __call__(self, cost: float) -> jnp.ndarray:
        """
        Returns the control based on current cost and internal parameters.
        """
        
        # 1. observe next state and update histories
        prev_histories = (self.cost_history, self.control_history)
        self.cost_history = append(self.cost_history, cost)
        self.control_history = append(self.control_history, self.prev_control)
        self.t += 1
        histories = (self.cost_history, self.control_history)
        state = self.lifter.map(*histories)  # xhat_{t+1}
        if self.t < self.T0: 
            lifter_loss = self.lifter.update(prev_histories, histories)  # update lifter, if needed
        else:
            if self.t == self.T0: print('WARNING: note that we are only updating lifter during sysid phase')
            lifter_loss = 0.
        
        # 2. explore for sysid, and then get stabilizing controller
        if self.t < self.T0:
            control = self.sysid.perturb_control(state)
            self.prev_state = state
            self.prev_control = control
            return control
        elif self.K_every is not None:  
            if self.t == self.T0 or (self.t - self.T0) % self.K_every == 0:  # get the K from sysid every so often
                print('copying the K from {}'.format(self.sysid))
                self.K = self.sysid.get_lqr()
                pass
        
        # 3. compute disturbance
        pred_state = self.sysid.dynamics(self.prev_state, self.prev_control)  # A @ xhat_t + B @ u_t
        disturbance = state - pred_state  # xhat_{t+1} - (A @ xhat_t + B @ u_t)
        self.disturbance_history = append(self.disturbance_history, disturbance)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        
        # compute change in cost, as well as the new scale
        cost_diff = cost - self.prev_cost
        M_scale, M0_scale, K_scale = map(lambda s: s / (self.t ** 0.25) if self.decay_scales else s, [self.M_scale, self.M0_scale, self.K_scale])

        # 4. update controller
        d = self.d_rescale_u(self.prev_control)
        grad_M = self.grad_M(cost_diff) * d.reshape(1, -1, 1)
        grad_M0 = self.grad_M0(cost_diff) * d.reshape(-1)
        grad_K = self.grad_K(cost_diff)
        self.grads.append((grad_M, grad_M0, grad_K))
        if len(self.grads) == self.grads.maxlen and self.t % self.step_every == 0:
            grads = list(self.grads)[:self.step_every]  # use updates starting from h steps ago
            self.M = self.M - self.M_update_rescaler.step(sum([g[0] for g in grads]), iterate=self.M)
            self.M0 = self.M0 - self.M0_update_rescaler.step(sum([g[1] for g in grads]), iterate=self.M0)
            self.K = self.K - self.K_update_rescaler.step(sum([g[2] for g in grads]), iterate=self.K)
        
        # 5. compute newest perturbed control
        M_tilde, M0_tilde, K_tilde = self.M, self.M0, self.K

        if self.method == 'FKM':  # perturb em all
            eps_M = sample(jkey(), (self.h, self.control_dim, self.state_dim))
            eps_M0 = sample(jkey(), (self.control_dim,))
            eps_K = sample(jkey(), (self.control_dim, self.state_dim))
            M_tilde = M_tilde + M_scale * eps_M
            M0_tilde = M0_tilde + M0_scale * eps_M0
            K_tilde = K_tilde + K_scale * eps_K
            if M_scale > 0: self.eps_M = append(self.eps_M, eps_M)
            if M0_scale > 0: self.eps_M0 = append(self.eps_M0, eps_M0)
            if K_scale > 0: self.eps_K = append(self.eps_K, eps_K)
            
        elif self.method == 'REINFORCE':  # perturb output only
            eps = sample(jkey(), (self.control_dim,))
            M0_tilde = M0_tilde + M0_scale * eps
            if M0_scale > 0: self.eps = append(self.eps, eps / M0_scale)
            
        control = self.inv_rescale_u(-K_tilde @ state) + M0_tilde + jnp.tensordot(M_tilde, self.disturbance_history[-self.h:], axes=([0, 2], [0, 1]))        
        control = self.rescale_u(control)
#         control = self.sysid.perturb_control(state, control=control)  # perturb for sysid purposes later than T0?

        # cache it
        self.prev_cost = cost
        self.prev_state = state
        self.prev_control = control 
        
        # update stats
        A, B = self.sysid.sysid()
        self.stats.update('||A-BK||_op', opnorm(A - B @ self.K), t=self.t)
        self.stats.update('||A||_op', opnorm(A), t=self.t)
        self.stats.update('||B||_F', jnp.linalg.norm(B, 'fro').item(), t=self.t)
        self.stats.update('disturbances', jnp.linalg.norm(disturbance).item(), t=self.t)
        self.stats.update('lifter losses', lifter_loss, t=self.t)
        if self.control_dim == 1:
            self.stats.update('K @ state', (-self.K @ state).item(), t=self.t)
            self.stats.update('M \cdot w', (jnp.tensordot(self.M, self.disturbance_history[-self.h:], axes=([0, 2], [0, 1]))).item(), t=self.t)
            self.stats.update('M0', self.M0.item(), t=self.t)
            
        return control
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        return self(cost)
    

# ---------------------------------------------------------------------------------------------------------------------

from deluca.agents._lqr import LQR as _LQR
from deluca.agents._gpc import GPC as _GPC
from deluca.agents._bpc import BPC as _BPC
    
class LQR(_LQR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = Stats()
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        return self(state)
    
class GPC(_GPC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = Stats()
        self.stats.register('K @ state', float, plottable=True)
        self.stats.register('M \cdot w', float, plottable=True)
        self.stats.register('M0', float, plottable=True)
        self.stats.register('disturbances', float, plottable=True)
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        if self.state.shape[0] == 1:
            self.stats.update('disturbances', self.noise_history[-1].item(), t=self.t)
        if self.action.shape[0] == 1:
            self.stats.update('K @ state', (-self.K @ self.state).item(), t=self.t)
            self.stats.update('M \cdot w', (jnp.tensordot(self.M, self.last_h_noises(), axes=([0, 2], [0, 1]))).item(), t=self.t)
            self.stats.update('M0', 0., t=self.t)
            
        return self(state)
    
class BPC(_BPC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = Stats()
        self.stats.register('K @ state', float, plottable=True)
        self.stats.register('M \cdot w', float, plottable=True)
        self.stats.register('M0', float, plottable=True)
        self.stats.register('disturbances', float, plottable=True)
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        if self.state.shape[0] == 1:
            self.stats.update('disturbances', self.noise_history[-1].item(), t=self.t)
        if self.action.shape[0] == 1:
            self.stats.update('K @ state', (-self.K @ self.state).item(), t=self.t)
            self.stats.update('M \cdot w', (jnp.tensordot(self.M, self.noise_history, axes=([0, 2], [0, 1]))).item(), t=self.t)
            self.stats.update('M0', 0., t=self.t)
            
        return self(state, cost)
    
class RBPC(Controller):  # from Udaya
    def __init__(self, A, B, Q, R, M, H, lr, delta, noise_sd):
        n, m = B.shape
        self.n, self.m = n, m
        self.lr, self.A, self.B, self.M, self.H, self.noise_sd = lr, A, B, M, H, noise_sd
        self.x, self.u, self.delta, self.t = jnp.zeros((n, 1)), jnp.zeros((m, 1)), delta, 0
        self.K, self.E, self.W = LQR(A, B, Q, R).K, jnp.zeros((M, n, m)), jnp.zeros((M + H -1, n))
        self.eps = sample(jkey(), (self.M, self.m), 'sphere')
        
        self.stats = Stats()
        self.stats.register('K @ state', float, plottable=True)
        self.stats.register('M \cdot w', float, plottable=True)
        self.stats.register('M0', float, plottable=True)
        self.stats.register('disturbances', float, plottable=True)
        pass

    def Egrad(self, cost):
        gE = jnp.zeros((self.M, self.n, self.m))
        for i in range(self.H):
          gE += jnp.einsum("ij, k->ijk",self.W[i:self.M+i,:], self.eps[i])
        return gE * cost

    def act(self, x, cost):
        # 1. Get gradient estimates
        delta_E = self.Egrad(cost)

        # 2. Execute updates
        self.E -= self.lr * delta_E

        # 3. Ensure norm is good
        norm = jnp.linalg.norm(self.E)
        if norm > 1:
           self.E *= 1/ norm

        # 4. Get new noise
        w = x - self.A @ self.x - self.B @ self.u
        w = w.reshape(self.n)
        self.W = self.W.at[0].set(w)
        self.W = jnp.roll(self.W, -1, axis = 0)
            
        # 5. Get new eps (after parameter update (4) or ...?)
        noise = sample(jkey(), (self.m,), 'sphere')
        self.eps = self.eps.at[0].set(self.noise_sd * noise)
        self.eps = jnp.roll(self.eps, -1, axis = 0)

        # 5. Update x & t and get action
        self.x, self.t = x, self.t + 1
        u = -self.K @ x + jnp.tensordot(self.E , self.W[-self.M:], axes = ([0, 1], [0, 1])).reshape((self.m, 1)) + self.eps[-1].reshape((self.m, 1))
        self.u = u.reshape((self.m, 1))
              
        return self.u.reshape(self.m)
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        if self.n == 1:
            self.stats.update('disturbances', self.W[-1].item(), t=self.t)
        if self.m == 1:
            self.stats.update('K @ state', (-self.K @ self.x).item(), t=self.t)
            self.stats.update('M \cdot w', (jnp.tensordot(self.E, self.W[-self.M:], axes=([0, 1], [0, 1]))).item(), t=self.t)
            self.stats.update('M0', 0., t=self.t)
        return self.act(state, cost)
    
# ----------------------------------------------------------------------------------------