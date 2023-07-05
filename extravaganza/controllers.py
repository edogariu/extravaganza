import logging
from abc import abstractmethod
from typing import Tuple
from collections import deque, defaultdict

import jax.numpy as jnp

from extravaganza.lifters import Lifter
from extravaganza.sysid import SysID
from extravaganza.stats import Stats
from extravaganza.utils import rescale, d_rescale, inv_rescale, append, sample, set_seed, jkey, opnorm, get_classname

class Controller:
    """
    abstract class for all controllers
    """
    control_dim: int = None
    
    @abstractmethod
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        """
        gets the next control as a function of the cost and arrived-in state from playing the previous control
        """
        pass


class LiftedBPC(Controller):
    """
    Naturally expected to output on the scale of `(-1, 1)`.
    If `bounds` is not `None`, it will automatically clip (or apply `tanh`) and rescale to the given bounds.
    """
    def __init__(self,
                 h: int,
                 initial_u: jnp.ndarray,
                 rescalers,
                 initial_scales: Tuple[float, float, float],
                 T0: int,
                 bounds = None,
                 method = 'REINFORCE',
                 lifter: Lifter = None,
                 sysid: SysID = None,
                 K: jnp.ndarray = None,
                 step_every: int = 1,
                 use_tanh = True,
                 decay_scales = False,
                 use_K_from_sysid: bool = False,
                 seed: int = None,
                 stats: Stats = None):

        set_seed(seed)  # for reproducibility
        
        # check things make sense
        if lifter is None and isinstance(sysid, Lifter): lifter = sysid
        if sysid is None and isinstance(lifter, SysID): sysid = lifter
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
        self.use_K_from_sysid = use_K_from_sysid
        self.initial_control = initial_u
        self.step_every = step_every

        # for rescaling controls
        self.rescale_u = lambda u: rescale(u, self.bounds, use_tanh=use_tanh) if self.bounds is not None else u
        self.inv_rescale_u = lambda ru: inv_rescale(ru, self.bounds, use_tanh=use_tanh) if self.bounds is not None else ru
        self.d_rescale_u = lambda u: d_rescale(u, self.bounds, use_tanh=use_tanh) if self.bounds is not None else jnp.ones_like(u)        
        
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
        self.state_history = jnp.zeros((2 * self.h, self.state_dim))
        self.cost_history = jnp.zeros(self.hh)  # for sysid/lifting
        self.control_history = jnp.zeros((self.hh, self.control_dim))  # for sysid/lifting
        self.cost_accum = 0.
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
                # val = self.eps[-1].reshape(self.control_dim, 1) @ self.prev_state.reshape(1, self.state_dim)
                val = sum([jnp.transpose(jnp.einsum('ij,k->ijk', self.state_history[i: self.h + i], self.eps[i]), axes=(0, 2, 1)) for i in range(self.h)]).sum(axis=0)
                return diff * val #* self.control_dim * self.state_dim * self.h
            
        self.grads = deque([(jnp.zeros_like(self.M), jnp.zeros_like(self.M0), jnp.zeros_like(self.K))], maxlen=self.h)
        self.grad_M = grad_M
        self.grad_M0 = grad_M0
        self.grad_K = grad_K
        
        self.M_update_rescaler = rescalers[0]()
        self.M0_update_rescaler = rescalers[1]()
        self.K_update_rescaler = rescalers[2]()
        
        # stats
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        self.stats.register('||A-BK||_op', float, plottable=True)
        self.stats.register('||A||_op', float, plottable=True)
        self.stats.register('||B||_F', float, plottable=True)
        self.stats.register('disturbances', float, plottable=True)
        self.stats.register('costs', float, plottable=True)
        self.stats.register('cost diffs', float, plottable=True)
        self.stats.register('avg costs', float, plottable=True)
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
        # cost = cost - self.cost_accum / self.t  # subtract running avg
        
        # 1. observe next state and update histories
        prev_histories = (self.cost_history, self.control_history)
        self.cost_history = append(self.cost_history, cost)
        self.control_history = append(self.control_history, self.prev_control)
        self.t += 1
        histories = (self.cost_history, self.control_history)
        state = self.lifter.map(*histories)  # xhat_{t+1}
        self.state_history = append(self.state_history, state)
        if self.t < self.T0: 
            lifter_loss = self.lifter.update(prev_histories, histories)  # update lifter, if needed
        else:
            if self.t == self.T0: logging.info('({}) Note that we are only updating lifter during sysid phase'.format(get_classname(self)))
            lifter_loss = 0.
        
        # 2. explore for sysid, and then maybe get stabilizing controller
        if self.t < self.T0:
            control = self.sysid.perturb_control(state)  # TODO should rescaling be applied to this?
            if self.bounds is not None: control = control.clip(*self.bounds)
            self.prev_state = state
            self.prev_control = control
            return control
        elif self.use_K_from_sysid and self.t == self.T0:  # get the K from sysid every so often
            logging.info('({}) copying the K from {}'.format(get_classname(self), self.sysid))
            self.K = self.sysid.get_lqr()
            pass
    
        # 3. compute disturbance
        pred_state = self.sysid.dynamics(self.prev_state, self.prev_control)  # A @ xhat_t + B @ u_t
        disturbance = state - pred_state  # xhat_{t+1} - (A @ xhat_t + B @ u_t)
        self.disturbance_history = append(self.disturbance_history, disturbance)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        
        # compute change in cost, as well as the new scale
        cost_diff = cost - self.prev_cost
        M_scale, M0_scale, K_scale = map(lambda s: s / ((self.t - self.T0 + 1) ** 0.25) if self.decay_scales else s, [self.M_scale, self.M0_scale, self.K_scale])

        # 4. update controller
        d = self.d_rescale_u(self.prev_control)
        grad_M = self.grad_M(cost_diff) * d.reshape(1, -1, 1)
        grad_M0 = self.grad_M0(cost_diff) * d.reshape(-1)
        grad_K = self.grad_K(cost_diff)
        self.grads.append((grad_M, grad_M0, grad_K))
        if len(self.grads) == self.grads.maxlen and self.t % self.step_every == 0:
            grads = list(self.grads)[:self.step_every]  # use updates starting from h steps ago
            self.M -= self.M_update_rescaler.step(sum([g[0] for g in grads]), iterate=self.M)
            self.M0 -= self.M0_update_rescaler.step(sum([g[1] for g in grads]), iterate=self.M0)
            self.K -= self.K_update_rescaler.step(sum([g[2] for g in grads]), iterate=self.K)
            
        # # ensure norms are good
        # norm = jnp.linalg.norm(self.M)  
        # if norm > 1:
        #    self.M /= norm
        # norm = jnp.linalg.norm(self.K)  
        # if norm > 1:
        #     self.K /= norm
        
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
            if M0_scale > 0: self.eps = append(self.eps, eps)
            
        # TODO this might not be the right thing to do with `K` when rescaling!!!!
        control = self.inv_rescale_u(-K_tilde @ state) + M0_tilde + jnp.tensordot(M_tilde, self.disturbance_history[-self.h:], axes=([0, 2], [0, 1]))        
        control = self.rescale_u(control)
        if self.bounds is not None: control = jnp.clip(control, *self.bounds)
#         control = self.sysid.perturb_control(state, control=control)  # perturb for sysid purposes later than T0?

        # cache it
        # self.prev_cost = cost
        self.prev_cost = jnp.mean(self.cost_history).item()
        self.prev_state = state
        self.prev_control = control 
        self.cost_accum += cost
        
        # update stats
        A, B = self.sysid.sysid()
        self.stats.update('||A-BK||_op', opnorm(A - B @ self.K), t=self.t)
        self.stats.update('||A||_op', opnorm(A), t=self.t)
        self.stats.update('||B||_F', jnp.linalg.norm(B, 'fro').item(), t=self.t)
        self.stats.update('disturbances', jnp.linalg.norm(disturbance).item(), t=self.t)
        self.stats.update('costs', cost, t=self.t)
        self.stats.update('cost diffs', cost_diff, t=self.t)
        self.stats.update('lifter losses', lifter_loss, t=self.t)
        if self.control_dim == 1:
            self.stats.update('K @ state', (-self.K @ state).item(), t=self.t)
            self.stats.update('M \cdot w', (jnp.tensordot(self.M, self.disturbance_history[-self.h:], axes=([0, 2], [0, 1]))).item(), t=self.t)
            self.stats.update('M0', self.M0.item(), t=self.t)
            
        return control
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        return self(cost)
    

# ---------------------------------------------------------------------------------------------------------------------
#         VARIOUS BASELINES
# ---------------------------------------------------------------------------------------------------------------------

from deluca.agents._lqr import LQR as _LQR
from deluca.agents._gpc import GPC as _GPC
from deluca.agents._bpc import BPC as _BPC
    
def get_seed(kwargs):
    if 'seed' in kwargs:
            seed = kwargs['seed']
            del kwargs['seed']
    else:
        seed = None
    return seed
    
class ConstantController(Controller):
    def __init__(self, value: float, control_dim: int, stats: Stats = None) -> None:
        super().__init__()
        self.value = float(value)
        self.control_dim = control_dim
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.full(shape=(self.control_dim,), fill_value=self.value)
    
    
class LQR(_LQR):
    def __init__(self, *args, **kwargs):
        set_seed(get_seed(kwargs))
        super().__init__(*args, **kwargs)
        self.control_dim = self.K.shape[0]
        self.stats = Stats()
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        return self(state)
    
class HINF(Controller):
    def __init__(self, 
                 A: jnp.ndarray,
                 B: jnp.ndarray,
                 Q: jnp.ndarray = None,
                 R: jnp.ndarray = None,
                 T: int = 100,
                 gamma: float = 10,
                 seed: int = None):

        set_seed(seed)
        dx, du = B.shape
        if Q is None: Q = jnp.identity(dx, dtype=jnp.float32)
        if R is None: R = jnp.identity(du, dtype=jnp.float32)
        P, K = [jnp.zeros((dx, dx)) for _ in range(T + 1)], [jnp.zeros((du, dx)) for _ in range(T)], 
        P[T] = Q
        for t in range(T - 1, -1, -1):
            Lambda = jnp.eye(dx) + (B @ jnp.linalg.inv(R) @ B.T - gamma ** -2 * jnp.eye(dx)) @ P[t + 1]
            P[t] = Q + A.T @ P[t + 1] @ jnp.linalg.pinv(Lambda) @ A
            K[t] = -jnp.linalg.inv(R) @ B.T @ P[t + 1] @ jnp.linalg.pinv(Lambda) @ A
        
        self.state_dim = dx
        self.control_dim = du
        self.K = K
        self.t = 0
        self.stats = Stats()
        pass
    
    def get_control(self, 
                    cost: float, 
                    state: jnp.ndarray) -> jnp.ndarray:
        assert state.shape == (self.state_dim,)
        ret = self.K[min(self.t, len(self.K) - 1)] @ state
        self.t += 1
        return ret
    
class GPC(_GPC):
    def __init__(self, *args, **kwargs):
        set_seed(get_seed(kwargs))
        super().__init__(*args, **kwargs)
        self.control_dim = self.K.shape[0]
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
        set_seed(get_seed(kwargs))
        super().__init__(*args, **kwargs)
        self.control_dim = self.K.shape[0]
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
    def __init__(self, A, B, Q, R, M, H, lr, delta, noise_sd, seed: int = None):
        set_seed(seed)
        n, m = B.shape
        self.n, self.m = n, m
        self.lr, self.A, self.B, self.M, self.H, self.noise_sd = lr, A, B, M, H, noise_sd
        self.x, self.u, self.delta, self.t = jnp.zeros((n, 1)), jnp.zeros((m, 1)), delta, 0
        self.K, self.E, self.W = LQR(A, B, Q, R).K, jnp.zeros((M, n, m)), jnp.zeros((M + H -1, n))
        self.eps = sample(jkey(), (self.M, self.m), 'sphere')
        self.cost = 0.
        self.control_dim = self.K.shape[0]
        
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
        delta_E = self.Egrad(cost - self.cost)
        self.cost - cost

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

class PID(Controller):
    def __init__(self, 
                 control_dim: int, 
                 setpoint: jnp.ndarray,
                 obs_mask: jnp.ndarray,
                 Kp: jnp.ndarray = None, 
                 Ki: jnp.ndarray = None, 
                 Kd: jnp.ndarray = None, 
                 stats: Stats = None):
        
        obs_dim = int(sum(obs_mask))
        if isinstance(setpoint, float): setpoint = jnp.array([setpoint])
        assert setpoint.shape == (obs_dim,)
        
        self.Kp, self.Ki, self.Kd = map(lambda t: jnp.array(t).reshape(control_dim, obs_dim) if t is not None else jnp.zeros((control_dim, obs_dim)), 
                                        [Kp, Ki, Kd])
        self.obs_dim = obs_dim
        self.control_dim = control_dim
        self.setpoint = setpoint
        self.obs_idxs = jnp.where(jnp.array(obs_mask) == 1)[0]
        
        self.p = jnp.zeros(self.obs_dim)  # keep track of current error
        self.i = jnp.zeros(self.obs_dim)  # keep track of accumulated error

        self.t = 0
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        if obs_dim == 1:
            self.stats.register('Kp', float, plottable=True)
            self.stats.register('Ki', float, plottable=True)
            self.stats.register('Kd', float, plottable=True)
            self.stats.register('P', float, plottable=True)
            self.stats.register('I', float, plottable=True)
            self.stats.register('D', float, plottable=True)
        pass
    
    def reset(self, seed: int = None):
        set_seed(seed)  # for reproducibility

        self.p = jnp.zeros(self.obs_dim)
        self.i = jnp.zeros(self.obs_dim)
        return self
    
    def get_control(self, 
                    cost: float, 
                    state: jnp.ndarray) -> jnp.ndarray:
        state = jnp.take(state, self.obs_idxs)
        assert state.shape == (self.obs_dim,), state.shape
        
        error = state - self.setpoint
        
        p, i, d = self.p, self.i, error - self.p
        control = self.Kp @ p + self.Ki @ i + self.Kd @ d
        
        self.p = error
        self.i += error
        self.t += 1
        
        if self.obs_dim == 1:
            self.stats.update('Kp', self.Kp.item(), t=self.t)
            self.stats.update('Ki', self.Ki.item(), t=self.t)
            self.stats.update('Kd', self.Kd.item(), t=self.t)
            self.stats.update('P', p.item(), t=self.t)
            self.stats.update('I', i.item(), t=self.t)
            self.stats.update('D', d.item(), t=self.t)
        return control

"""
Regular clipped PPO implementation, with no target networks, no delayed updates, and no experience replay.
"""
import numpy as np
import torch
import torch.nn as nn
class PPO(Controller):
    def __init__(self,
                 state_dim: int, 
                 control_dim: int,  # n_actions if discrete
                 lr_actor: float, 
                 lr_critic: float, 
                 gamma: float, 
                 eps_clip: float, 
                 has_continuous_action_space: bool,
                 stats: Stats = None):
        self.state_dim = state_dim
        self.action_dim = control_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.continuous_action_space = has_continuous_action_space
        
        # models and opts
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.action_dim),
            nn.Softmax(dim=-1) if not has_continuous_action_space else nn.Identity()
        )
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # things to keep track of
        self.prob_act = None
        self.state = None
        self.cost = None
        
        self.t = 0
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        self.stats.register('lr_actor', float, plottable=True)
        self.stats.register('lr_critic', float, plottable=True)
        self.stats.register('gamma', float, plottable=True)
        self.stats.register('eps_clip', float, plottable=True)
        pass
    
    def reset(self, seed: int = None):
        set_seed(seed)  # for reproducibility
        self.prob_act = None
        self.state = None
        self.cost = None

        # reset models
        for model in [self.actor, self.critic]:
            for layer in model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            model.train()
        for opt in [self.actor_opt, self.critic_opt]:
            opt.__setstate__({'state': defaultdict(dict)}) 
        return self
    
    def policy_loss(self, old_log_prob, log_prob, advantage, eps):
        ratio = (log_prob - old_log_prob).exp()
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        m = torch.min(ratio * advantage, clipped * advantage)
        return -m
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        
        # compute next action
        if state != 'done':
            state = torch.from_numpy(np.array(state))
            done = False
        else:
            state = torch.zeros(self.state_dim)
            done = True
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        prob_act = dist.log_prob(action)
        
        # update
        if self.prob_act is not None:
            reward = self.cost - cost  # because `cost = -\sum_i reward_i`
            advantage = reward + (1 - done) * self.gamma * self.critic(state) - self.critic(torch.from_numpy(self.state))
            
            actor_loss = self.policy_loss(self.prob_act.detach(), prob_act, advantage.detach(), self.eps_clip)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            critic_loss = advantage.pow(2).mean()
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
        
        self.state = state
        self.prob_act = prob_act
        self.cost = cost
        
        self.t += 1
        self.stats.update('lr_actor', self.ppo.actor_opt.param_groups[0]['lr'], t=self.t)
        self.stats.update('lr_critic', self.ppo.critic_opt.param_groups[0]['lr'], t=self.t)
        self.stats.update('gamma', self.ppo.gamma, t=self.t)
        self.stats.update('eps_clip', self.ppo.eps_clip, t=self.t)
        
        return jnp.array(action.detach().data.numpy())
    