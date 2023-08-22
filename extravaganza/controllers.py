import logging
from abc import abstractmethod
from typing import Tuple
from collections import deque

import jax.numpy as jnp

from extravaganza.stats import Stats
from extravaganza.utils import rescale, d_rescale, inv_rescale, append, sample, set_seed, jkey, opnorm, get_classname, dare_gain

BETA = 1e6  # norm clipping for both the controller matrices and their grads

def clip(x: jnp.ndarray, name: str = None, beta: float = BETA):
    norm = jnp.linalg.norm(x)  
    if norm > beta:
        if name is not None: logging.info('(CONTROLLER): clipped {}!'.format(name))
        x /= (norm / beta)
    return x
    

class Controller:
    """
    abstract class for all controllers
    """
    control_dim: int = None
    state_dim: int = None
    stats: Stats = None
    
    @abstractmethod
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        """
        gets the next control as a function of the cost and arrived-in observation from playing the previous control (and possibly the disturbance given by the arrived-in observation)
        """
        pass
                           
    def update(self, 
               state: jnp.ndarray, cost: float,   # state we were in 
               control: jnp.ndarray,  # control we played
               next_state: jnp.ndarray, next_cost: float,  # state we landed in
               ):
        """
        Function to update the controller, by default a no-op
        """
        pass
    
    def system_reset_hook(self):
        """
        This is a function that gets called every time the dynamical system we are controlling gets episodically reset.
        For convenience only, by default a no-op
        """
        pass


class EvanBPC(Controller):
    """
    Naturally expected to output on the scale of `(-1, 1)`.
    If `bounds` is not `None`, it will automatically clip (or apply `tanh`) and rescale to the given bounds.
    """
    def __init__(self,
                 A: jnp.ndarray,
                 B: jnp.ndarray,
                 h: int,
                 initial_u: jnp.ndarray,
                 rescalers,
                 initial_scales: Tuple[float, float, float],
                 bounds = None,
                 method = 'REINFORCE',
                 use_tanh = False,
                 decay_scales = False,
                 use_stabilizing_K: bool = False,
                 seed: int = None,
                 stats: Stats = None):

        set_seed(seed)  # for reproducibility
        
        # check things make sense
        self.state_dim, self.control_dim = B.shape
        
        assert initial_u.shape[0] == self.control_dim
        assert method in ['FKM', 'REINFORCE']
        assert all(map(lambda i: i >= 0, initial_scales))
        if bounds is not None:
            bounds = jnp.array(bounds).reshape(2, -1)
            assert len(bounds[0]) == len(bounds[1]) and len(bounds[0]) == self.control_dim, 'improper bounds'
            assert all(map(lambda i: bounds[0, i] < bounds[1, i], range(self.control_dim))), 'improper bounds'
        
        self.A, self.B = A, B
        
        # hyperparams
        self.h = h
        self.method = method
        self.bounds = bounds
        self.decay_scales = decay_scales
        self.initial_control = initial_u
        self.reset_t = -1

        # for rescaling controls
        self.rescale_u = lambda u: rescale(u, self.bounds, use_tanh=use_tanh) if self.bounds is not None else u
        self.inv_rescale_u = lambda ru: inv_rescale(ru, self.bounds, use_tanh=use_tanh) if self.bounds is not None else ru
        self.d_rescale_u = lambda u: d_rescale(u, self.bounds, use_tanh=use_tanh) if self.bounds is not None else jnp.ones_like(u)        
        
        # controller params
        self.M = jnp.zeros((self.h, self.control_dim, self.state_dim))
        self.M0 = self.inv_rescale_u(initial_u)
        self.K = jnp.zeros((self.control_dim, self.state_dim))
        if use_stabilizing_K:  # initialize state feedback controller as LQR solution to stabilize system
            K = dare_gain(self.A, self.B)
            oppy = opnorm(self.A - self.B @ K)
            if oppy < 1:
                logging.info('(CONTROLLER): we WILL be using the stabilizing controller with ||A-BK||_op={}'.format(oppy))
                self.K = K
            else: logging.warning('(CONTROLLER): we will NOT be using the stabilizing controller with ||A-BK||_op={}'.format(oppy))
        self.M_scale, self.M0_scale, self.K_scale = initial_scales
        
        # histories are rightmost recent (increasing in time)
        self.costs = deque(maxlen=2 * self.h)
        self.prev_control = jnp.zeros(self.control_dim)
        self.disturbance_history = jnp.zeros((2 * self.h, self.state_dim))  # past 2h disturbances, for controller
        self.state_history = jnp.zeros((2 * self.h, self.state_dim))
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
            
            def grad_M(diff) -> jnp.ndarray:
                val = sum([jnp.transpose(jnp.einsum('ij,k->ijk', self.disturbance_history[i: self.h + i], self.eps[i]), axes=(0, 2, 1)) for i in range(self.h)])
                return diff * val #* self.control_dim * self.state_dim * self.h
            def grad_M0(diff) -> jnp.ndarray:
                return diff * self.eps[-1] #* self.control_dim * self.state_dim * self.h
            def grad_K(diff) -> jnp.ndarray:
                # val = self.eps[-1].reshape(self.control_dim, 1) @ self.prev_state.reshape(1, self.state_dim)
                val = sum([jnp.transpose(jnp.einsum('ij,k->ijk', self.state_history[i: self.h + i], self.eps[i]), axes=(0, 2, 1)) for i in range(self.h)]).sum(axis=0)
                return -diff * val #* self.control_dim * self.state_dim * self.h
            
        self.grads = deque([(jnp.zeros_like(self.M), jnp.zeros_like(self.M0), jnp.zeros_like(self.K))], maxlen=self.h)
        self.grad_M = grad_M
        self.grad_M0 = grad_M0
        self.grad_K = grad_K
        
        self.M_update_rescaler = rescalers[0]()
        self.M0_update_rescaler = rescalers[1]()
        self.K_update_rescaler = rescalers[2]()
        
        # stats
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so we will use the sysid states.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        self.stats.register('states', obj_class=jnp.ndarray, shape=(self.state_dim,))
        self.stats.register('disturbances', obj_class=jnp.ndarray, shape=(self.state_dim,))
        self.stats.register('-K @ state', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('M \cdot w', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('M0', obj_class=jnp.ndarray, shape=(self.control_dim,))
        pass

# ------------------------------------------------------------------------------------------------------------
    
    def system_reset_hook(self):
        self.reset_t = self.t
        return super().system_reset_hook()
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        assert state.shape == (self.state_dim,)
        
        M_scale, M0_scale, K_scale = map(lambda s: s / (self.t ** 0.25) if self.decay_scales else s, [self.M_scale, self.M0_scale, self.K_scale])
        M_tilde, M0_tilde, K_tilde = self.M, self.M0, self.K

        if self.method == 'FKM':  # perturb em all
            eps_M = sample(jkey(), (self.h, self.control_dim, self.state_dim), 'normal')
            eps_M0 = sample(jkey(), (self.control_dim,), 'normal')
            eps_K = sample(jkey(), (self.control_dim, self.state_dim), 'normal')
            M_tilde = M_tilde + M_scale * eps_M
            M0_tilde = M0_tilde + M0_scale * eps_M0
            K_tilde = K_tilde + K_scale * eps_K
            if M_scale > 0: self.eps_M = append(self.eps_M, eps_M)
            if M0_scale > 0: self.eps_M0 = append(self.eps_M0, eps_M0)
            if K_scale > 0: self.eps_K = append(self.eps_K, eps_K)
            
        elif self.method == 'REINFORCE':  # perturb output only
            eps = sample(jkey(), (self.control_dim,), 'normal')
            M0_tilde = M0_tilde + M0_scale * eps
            if M0_scale > 0: self.eps = append(self.eps, eps)
            
        # TODO this might not be the right thing to do with `K` when rescaling!!!!
        control = -K_tilde @ state + M0_tilde + jnp.tensordot(M_tilde, self.disturbance_history[-self.h:], axes=([0, 2], [0, 1]))        
        control = self.rescale_u(control)
        if self.bounds is not None: 
            for i in range(self.control_dim): control = control.at[i].set(jnp.clip(control[i], *self.bounds[i]))
            # control = jnp.clip(control, *self.bounds)
        
        # update stats
        self.stats.update('states', state, t=self.t)
        self.stats.update('-K @ state', (-self.K @ state).reshape(self.control_dim), t=self.t)
        self.stats.update('M \cdot w', (jnp.tensordot(self.M, self.disturbance_history[-self.h:], axes=([0, 2], [0, 1]))).reshape(self.control_dim), t=self.t)
        self.stats.update('M0', self.M0, t=self.t)
        return jnp.clip(control, -1e8, 1e8)
    
    
    def update(self, state: jnp.ndarray, cost: float, control: jnp.ndarray, next_state: jnp.ndarray, next_cost: float):
        assert state.shape == next_state.shape and state.shape == (self.state_dim,)
        assert control.shape == (self.control_dim,)
                
        self.t += 1
        
        # 1. observe state and disturbance
        disturbance = next_state - (self.A @ state + self.B @ control) if self.reset_t == -1 else jnp.zeros(self.state_dim)
        self.state_history = append(self.state_history, next_state)
        self.disturbance_history = append(self.disturbance_history, disturbance)
        
        if self.reset_t > 0: 
            if self.t - self.reset_t <= self.h + 1: return  # don't update controller for `h` steps after resets
            else: self.reset_t = -1 
        
        # 2. compute change in cost for zero order optimization
        cost_diff = next_cost - cost

        # 3. update controller
        d = self.d_rescale_u(control)
        grad_M = self.grad_M(cost_diff) * d.reshape(1, -1, 1)
        grad_M0 = self.grad_M0(cost_diff) * d.reshape(-1)
        grad_K = self.grad_K(cost_diff) * d.reshape(-1, 1)
        grad_M, grad_M0, grad_K = map(lambda arr: clip(arr), (grad_M, grad_M0, grad_K))  # use updates starting from h steps ago via popping left
        self.grads.append((grad_M, grad_M0, grad_K))
        if len(self.grads) == self.grads.maxlen:
            grad_M, grad_M0, grad_K = self.grads.popleft()  # use updates starting from h steps ago via popping left
            self.M -= self.M_update_rescaler.step(grad_M, iterate=self.M)
            self.M0 -= self.M0_update_rescaler.step(grad_M0, iterate=self.M0)
            self.K -= self.K_update_rescaler.step(grad_K, iterate=self.K)  # note that this is minus since  u = -K @ x + ...
            
        # 4. ensure norms are good
        self.M = clip(self.M, 'M')
        self.K = clip(self.K, 'K')
        self.M0 = clip(self.M0, 'M0')
            
        self.stats.update('disturbances', disturbance, t=self.t)
        pass
        
    
# ---------------------------------------------------------------------------------------------------------------------
#         VARIOUS BASELINES AND OTHER CONTROLLERS
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
    def __init__(self, value: jnp.ndarray, state_dim: int, stats: Stats = None) -> None:
        super().__init__()
        self.value = value
        if not isinstance(self.value, jnp.ndarray):
            self.value = jnp.array(self.value)
        self.control_dim = value.shape[0]
        self.state_dim = state_dim
        
        self.t = 0
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        self.stats.register('states', obj_class=jnp.ndarray, shape=(self.state_dim,))
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        self.stats.update('states', state, t=self.t)
        self.t += 1
        return self.value
    
    
class LQR(_LQR):
    def __init__(self, *args, **kwargs):
        set_seed(get_seed(kwargs))
        super().__init__(*args, **kwargs)
        self.control_dim, self.state_dim = self.K.shape

        self.stats = Stats()
        self.t = 0
        self.stats.register('states', obj_class=jnp.ndarray, shape=(self.state_dim,))
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        assert state.shape == (self.state_dim,), (state.shape, self.state_dim)
        self.t += 1
        self.stats.update('states', state, t=self.t)
        return self(state)
    
    def system_reset_hook(self): pass
    def update(self, *args, **kwargs): pass
    
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
        self.stats.register('states', obj_class=jnp.ndarray, shape=(self.state_dim,))
        pass
    
    def get_control(self, 
                    cost: float, 
                    state: jnp.ndarray) -> jnp.ndarray:
        assert state.shape == (self.state_dim,), (state.shape, self.state_dim)
        ret = self.K[min(self.t, len(self.K) - 1)] @ state
        self.stats.update('states', state, t=self.t)
        self.t += 1
        return ret
    
class GPC(_GPC):
    def __init__(self, *args, **kwargs):
        set_seed(get_seed(kwargs))
        super().__init__(*args, **kwargs)
        self.control_dim, self.state_dim = self.K.shape
        self.stats = Stats()
        self.stats.register('states', obj_class=jnp.ndarray, shape=(self.state_dim,))
        self.stats.register('-K @ state', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('M \cdot w', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('M0', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('disturbances', obj_class=jnp.ndarray, shape=(self.state_dim,))
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        assert state.shape == (self.state_dim,), (state.shape, self.state_dim)
        
        self.stats.update('states', self.state.reshape(self.state_dim), t=self.t)
        self.stats.update('-K @ state', (-self.K @ self.state).reshape(self.control_dim), t=self.t)
        self.stats.update('M \cdot w', (jnp.tensordot(self.M, self.last_h_noises(), axes=([0, 2], [0, 1]))).reshape(self.control_dim), t=self.t)
        self.stats.update('M0', jnp.zeros(self.control_dim), t=self.t)
        self.stats.update('disturbances', self.noise_history[-1].reshape(self.state_dim), t=self.t)
        assert state.ndim == 1, state.shape
            
        return self(state)
    
    def system_reset_hook(self): pass
    def update(self, *args, **kwargs): pass

class BPC(_BPC):
    def __init__(self, *args, **kwargs):
        set_seed(get_seed(kwargs))
        super().__init__(*args, **kwargs)
        self.control_dim, self.state_dim = self.K.shape
        self.stats = Stats()
        self.stats.register('states', obj_class=jnp.ndarray, shape=(self.state_dim,))
        self.stats.register('-K @ state', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('M \cdot w', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('M0', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('disturbances', obj_class=jnp.ndarray, shape=(self.state_dim,))
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        assert state.shape == (self.state_dim,), (state.shape, self.state_dim)

        self.stats.update('states', self.state.reshape(self.state_dim), t=self.t)
        self.stats.update('-K @ state', (-self.K @ self.state).reshape(self.control_dim), t=self.t)
        self.stats.update('M \cdot w', (jnp.tensordot(self.M, self.noise_history, axes=([0, 2], [0, 1]))).reshape(self.control_dim), t=self.t)
        self.stats.update('M0', jnp.zeros(self.control_dim), t=self.t)
        self.stats.update('disturbances', self.noise_history[-1].reshape(self.state_dim), t=self.t)
            
        return self(state.reshape(self.state_dim, 1), cost).reshape(self.control_dim)
    
    def system_reset_hook(self): pass
    def update(self, *args, **kwargs): pass
 
class RBPC(Controller):  # from Udaya
    def __init__(self, A, B, Q=None, R=None, M=5, H=5, lr=0.01, delta=0.001, noise_sd=0.05, seed: int = None):
        set_seed(seed)
        n, m = B.shape
        if Q is None: Q = jnp.eye(n)
        if R is None: R = jnp.eye(m)
        self.n, self.m = n, m
        self.lr, self.A, self.B, self.M, self.H, self.noise_sd = lr, A, B, M, H, noise_sd
        self.x, self.u, self.delta = jnp.zeros((n, 1)), jnp.zeros((m, 1)), delta
        self.K, self.E, self.W = LQR(A, B, Q, R).K, jnp.zeros((M, n, m)), jnp.zeros((M + H -1, n))
        self.eps = sample(jkey(), (self.M, self.m), 'sphere')
        self.cost = 0.
        self.control_dim, self.state_dim = self.K.shape
        
        self.stats = Stats()
        self.stats.register('states', obj_class=jnp.ndarray, shape=(self.state_dim,))
        self.stats.register('-K @ state', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('M \cdot w', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('M0', obj_class=jnp.ndarray, shape=(self.control_dim,))
        self.stats.register('disturbances', obj_class=jnp.ndarray, shape=(self.state_dim,))
        pass

    def Egrad(self, cost):
        gE = jnp.zeros((self.M, self.n, self.m))
        for i in range(self.H):
          gE += jnp.einsum("ij, k->ijk",self.W[i:self.M+i,:], self.eps[i])
        return gE * cost

    def act(self, x, cost):
        x = x.reshape(self.n, 1)
        
        # 1. Get gradient estimates
        delta_E = self.Egrad(cost - self.cost)
        self.cost = cost

        # 2. Execute updates
        self.E -= self.lr * delta_E

        # # 3. Ensure norm is good
        # norm = jnp.linalg.norm(self.E)
        # if norm > 1:
        #    self.E *= 1/ norm

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
        assert state.shape == (self.state_dim,), (state.shape, self.state_dim)
        
        self.stats.update('states', self.x.reshape(self.state_dim), t=self.t)
        self.stats.update('-K @ state', (-self.K @ self.x).reshape(self.control_dim), t=self.t)
        self.stats.update('M \cdot w', (jnp.tensordot(self.E, self.W[-self.M:], axes=([0, 1], [0, 1]))).reshape(self.control_dim), t=self.t)
        self.stats.update('M0', jnp.zeros(self.control_dim), t=self.t)
        self.stats.update('disturbances', self.W[-1].reshape(self.state_dim), t=self.t)
        
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
        
        state_dim = int(sum(obs_mask))
        if isinstance(setpoint, float): setpoint = jnp.array([setpoint])
        assert setpoint.shape == (state_dim,)
        self.Kp, self.Ki, self.Kd = map(lambda t: jnp.array(t).reshape(control_dim, state_dim) if t is not None else jnp.zeros((control_dim, state_dim)), 
                                        [Kp, Ki, Kd])
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.setpoint = setpoint
        self.obs_idxs = jnp.where(jnp.array(obs_mask) == 1)[0]
        
        self.p = jnp.zeros(self.state_dim)  # keep track of current error
        self.i = jnp.zeros(self.state_dim)  # keep track of accumulated error

        self.t = 0
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        self.stats.register('states', obj_class=jnp.ndarray, shape=(state_dim,))
        self.stats.register('Kp', obj_class=jnp.ndarray, shape=(control_dim * state_dim,))
        self.stats.register('Ki', obj_class=jnp.ndarray, shape=(control_dim * state_dim,))
        self.stats.register('Kd', obj_class=jnp.ndarray, shape=(control_dim * state_dim,))
        self.stats.register('P', obj_class=jnp.ndarray, shape=(state_dim,))
        self.stats.register('I', obj_class=jnp.ndarray, shape=(state_dim,))
        self.stats.register('D', obj_class=jnp.ndarray, shape=(state_dim,))
        pass
    
    def reset(self, seed: int = None):
        set_seed(seed)  # for reproducibility

        self.p = jnp.zeros(self.state_dim)
        self.i = jnp.zeros(self.state_dim)
        return self
    
    def get_control(self, 
                    cost: float, 
                    state: jnp.ndarray) -> jnp.ndarray:
        state = jnp.take(state, self.obs_idxs)
        assert state.shape == (self.state_dim,), (state.shape, self.state_dim)
        
        error = state - self.setpoint
        
        p, i, d = self.p, self.i, error - self.p
        control = self.Kp @ p + self.Ki @ i + self.Kd @ d
        
        self.p = error
        self.i += error
        self.t += 1
        
        self.stats.update('states', state, t=self.t)
        self.stats.update('Kp', self.Kp.reshape(-1), t=self.t)
        self.stats.update('Ki', self.Ki.reshape(-1), t=self.t)
        self.stats.update('Kd', self.Kd.reshape(-1), t=self.t)
        self.stats.update('P', p, t=self.t)
        self.stats.update('I', i, t=self.t)
        self.stats.update('D', d, t=self.t)
            
        return control.reshape(self.control_dim)

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
        self.exp_counter = 0  # for a random exploration phase
        self.prob_act = None
        self.state = None
        self.cost = None
        
        self.t = 0
        if stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            stats = Stats()
        self.stats = stats
        self.stats.register('lr actor', obj_class=float, plottable=True)
        self.stats.register('lr critic', obj_class=float, plottable=True)
        self.stats.register('gamma', obj_class=float, plottable=True)
        self.stats.register('eps clip', obj_class=float, plottable=True)
        pass
    
    def reset(self, seed: int = None):
        set_seed(seed)  # for reproducibility
        self.exp_counter = 0  # for a random exploration phase
        self.prob_act = None
        self.state = None
        self.cost = None

        # # reset models
        # for model in [self.actor, self.critic]:
        #     for layer in model.modules():
        #         if hasattr(layer, 'reset_parameters'):
        #             layer.reset_parameters()
        #     model.train()
        # for opt in [self.actor_opt, self.critic_opt]:
        #     opt.__setstate__({'state': defaultdict(dict)}) 
        return self
    
    def policy_loss(self, old_log_prob, log_prob, advantage, eps):
        ratio = (log_prob - old_log_prob).exp()
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
        m = torch.min(ratio * advantage, clipped * advantage)
        return -m
    
    def get_control(self, cost: float, obs: jnp.ndarray) -> jnp.ndarray:
        assert obs.shape == (self.state_dim,), 'PPO requires full observation'
        state = obs
        
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
        if self.exp_counter > 1:
            reward = self.cost - cost  # because `cost = -\sum_i reward_i`
            advantage = reward + (1 - done) * self.gamma * self.critic(state) - self.critic(self.state)
            
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
        self.exp_counter += 1
        
        self.t += 1
        self.stats.update('lr actor', self.actor_opt.param_groups[0]['lr'], t=self.t)
        self.stats.update('lr critic', self.critic_opt.param_groups[0]['lr'], t=self.t)
        self.stats.update('gamma', self.gamma, t=self.t)
        self.stats.update('eps clip', self.eps_clip, t=self.t)
        
        return jnp.array(action.detach().data.numpy())
    
class LambdaController(Controller):
    def __init__(self, controller: Controller, init_fn, get_control):
        super().__init__()
        self._controller = controller
        self.control_dim = self._controller.control_dim
        self._get_control = get_control
        init_fn(self)
        self.stats = self._controller.stats
        pass
    
    def get_control(self, cost: float, state: jnp.ndarray) -> jnp.ndarray:
        control = self._get_control(self, cost, state)
        return control
    