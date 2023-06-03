from typing import Tuple
from collections import deque
import numpy as np

from wrappers import FloatWrapper

TRANSFORMATION = 'w_rescale'  # ['sigmoid', 'rescale', 'w_sigmoid', 'w_rescale']

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def inv_sigmoid(s):
    return np.log(s / (1 - s))

def d_sigmoid(t):
    s = sigmoid(t)
    return s * (1 - s)

def rescale(t, interval):
    """
    for rescaling from `(-inf, inf) -> (a, b)`
    """
    a, b = interval
    if 'sigmoid' in TRANSFORMATION:
        t = sigmoid(t)  # t \in [0, 1]
    elif 'rescale' in TRANSFORMATION:
        t /= max(1, abs(t))  # t \in [-1, 1]
        t = (t + 1) / 2  # t \in [0, 1]
    t = a + t * (b - a)  # t \in [a, b]
    return t

def inv_rescale(s, interval):
    """
    for rescaling from `(a, b) -> (-inf, inf)` (ish, the scale shift method isnt invertible)
    """
    a, b = interval
    if 'sigmoid' in TRANSFORMATION:
        s = (s - a) / (b - a)
        t = inv_sigmoid(s)
    elif 'rescale' in TRANSFORMATION:
        t = 2 * (s - a) / (b - a) - 1
    return t

def d_rescale(t, interval):
    a, b = interval
    if 'sigmoid' in TRANSFORMATION:
        d = (b - a) * d_sigmoid(t)
    elif 'rescale' in TRANSFORMATION:
        d = (b - a) * (1 / max(1, abs(t))) / 2
    return d

def rescale_11(t, interval):
    """
    for rescaling from `(-1, 1) -> (a, b)`
    """
    a, b = interval
    t = (t + 1) / 2  # t \in [0, 1]
    t = a + t * (b - a)  # t \in [a, b]
    return t

def inv_rescale_11(s, interval):
    """
    for rescaling from `(a, b) -> (-1, 1)`
    """
    a, b = interval
    s = (s - a) / (b - a)  # s \in [0, 1]
    s = 2 * s - 1  # s \in [-1, 1]
    return s

def d_rescale_11(t, interval):
    a, b = interval
    return (b - a) / 2

class FloatHyperparameter(FloatWrapper):
    def __init__(self, 
                 h: int, 
                 initial_value: float, 
                 interval: Tuple[float, float]=(-float('inf'), float('inf')),
                 w_clip_size: float=float('inf'), 
                 M_clip_size: float=float('inf'), 
                 B_clip_size: float=float('inf'),
                 cost_clip_size: float=float('inf'), 
                 update_clip_size: float=float('inf'),
                 quadratic_term: float=0,
                 rescale: bool=False,
                 method='FKM'):
        """
        GPC for a float hyperparameter. 

        Parameters
        ----------
        h : float
            window size
        initial_value : float
            initial value for the hyperparameter to take
        interval : Tuple[float, float]
            bounds the hyperparameter can take
        w_clip_size : float
        M_clip_size : float
        B_clip_size : float
        cost_clip_size : float
        update_clip_size : float
        quadratic_term : float
        rescale : bool
        method : str
        """
        
        assert method in ['FKM', 'REINFORCE']
        
        self.h = h
        self.w_clip_size = w_clip_size
        self.M_clip_size = M_clip_size
        self.B_clip_size = B_clip_size
        self.cost_clip_size = cost_clip_size
        self.update_clip_size = update_clip_size
        self.quadratic_term = quadratic_term
        self.rescale_u = rescale and TRANSFORMATION[:2] != 'w_'
        self.rescale_w = rescale and TRANSFORMATION[:2] == 'w_'
        self.method = method
        
        if self.rescale_u:
            super().__init__(inv_rescale(initial_value, interval))
        elif self.rescale_w:
            super().__init__(inv_rescale_11(initial_value, interval))
        else:
            super().__init__(initial_value)
            
        self.interval = interval
        
        # dynamic parameters of the optimization
        # for meta lr ADAM
        self.m = np.zeros(self.h + 1)  # first order estimate
        self.v = np.zeros(self.h + 1)  # second order estimate
        self.meta_lr = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        
        self.M = np.zeros(self.h + 1)  # self.M[i] = M_i^t, but self.M[-1] is the bias
        self.M[-1] = self.value[0]
        self.prev_obj = None  # previous cost f(x_t)
        self.prev_value = self.get_value()  # previous value u_{t-1}
        self.t = 1  # timestep t
        self.B = None  # keep track of system estimates B_t, if needed
        self.r = None  # random variable to estimate system parameter B, if needed
        self.scale = None  # keep track of current scale G_t
        self.sigma = None
        if method == 'FKM':
            self.ns = np.zeros(self.h + 1)  # self.ns[i] = n_{t, i}, but self.ns[-1] = n_{t, 0} is the bias noise
            self.ws = deque([0 for _ in range(self.h)], maxlen=self.h)  # self.ws[i] = epsilon_{t-i-1}
        elif method == 'REINFORCE':
            self.ns = deque([0 for _ in range(self.h + 1)], maxlen=self.h + 1)  # self.ns[i] = n_{t-i}
            self.ws = deque([0 for _ in range(2 * self.h)], maxlen=2 * self.h)  # self.ws[i] = epsilon_{t-i-1}
        else:
            raise NotImplementedError()
    
    def get_value(self) -> float:  # rescale u_t if needed
        if self.rescale_u:
            return rescale(super().get_value(), self.interval)
        elif self.rescale_w:
            return rescale_11(super().get_value(), self.interval)
        else:
            return super().get_value()
    
    def step(self, obj: float, B: float=None, grad_u: float=None):
        """
        steps the controller once.
        FIRST STEP DOES NOTHING EXCEPT CACHE PREV OBSERVATION

        Parameters
        ----------
        obj : float
            `o_t`, `s_t`, `f(x_t)`, these are all the same
        B : float
            `B` param of the LDS
        grad_u : float
            gradient of cost w.r.t. control variable `u_t`
        """   
        
        # clip what we gotta
        obj = np.clip(obj, -self.cost_clip_size, self.cost_clip_size)
        if B is not None: B = np.clip(B, -self.B_clip_size, self.B_clip_size)
        
        # cache observation on first step
        if self.prev_obj is None:
            if obj == 0: return  
            self.prev_obj = obj
            self.scale = abs(obj)
            self.sigma = min(0.5, self.scale)
            if B is None:
                self.r = np.random.randn() * self.sigma
                self.B = self.r * obj
            return        
        
        # observe
        if B is None:
            b_lr = 1 / self.t
            self.B = (1 - b_lr) * self.B + b_lr * obj * self.r / self.sigma
            self.B = np.clip(self.B, -self.B_clip_size, self.B_clip_size)
            B = self.B
        pred = self.prev_obj + self.prev_value * B  # \hat{s_t} = f(x_{t-1}) + B
        w = obj - pred - self.quadratic_term * self.prev_value ** 2  # w_t = s_t - \hat{s_t} (maybe with - u_{t-1}^2)
        self.prev_obj = obj
                
        # calc gradients
        grad = np.zeros(self.h + 1)
        if grad_u is not None:
            grad[:-1] = grad_u * np.array(self.ws)[:self.h]
            grad[-1] = grad_u
        else:
            if self.method == 'FKM':
                grad = self.ns * obj / self.sigma
            elif self.method == 'REINFORCE':
                grad[:-1] = np.array(self.ns)[1:] * np.array(self.ws)[:self.h] * obj / self.sigma
                grad[-1] = self.ns[0] * obj / self.sigma
        if self.rescale_u: grad *= d_rescale(float(self.value[0]), self.interval)
        elif self.rescale_w: grad *= d_rescale_11(float(self.value[0]), self.interval)
        
        # compute scale and update
        scale_lr = 0.01# 1 / (self.t ** 0.5)
        self.scale = (1 - scale_lr) * self.scale + scale_lr * abs(obj)
        # self.scale = max(self.scale, abs(obj))
        self.sigma = min(0.01, self.scale / (self.t ** 0.25))
        update = self._compute_update(grad)
        update = np.clip(update, -self.update_clip_size, self.update_clip_size)
        self.M -= update
                
        # clip some more
        w = np.clip(w, -self.w_clip_size, self.w_clip_size)
        if self.rescale_w: 
            w = w / np.maximum(1, np.abs(w))
            v = 0.5
            # w_scale = self.scale
            # w = rescale(w, [-w_scale, w_scale])  # w \in [-w_scale, w_scale]
            self.M[:-1] = np.clip(self.M[:-1], -v / self.h, v / self.h)  # M \cdot w \in [-v, v]
            self.M[-1] = np.clip(self.M[-1], -(1 - v), 1 - v)
        else:
            # w = np.clip(w, -self.scale, self.scale)
            self.M[:-1] = np.clip(self.M[:-1], -self.M_clip_size, self.M_clip_size)
        
        # print('u_t={:.4f} \tG_t={:.4f} \t||grad||={:.4f} \tsigma_t={:.4f}\tB={:.4f}'.format(self.get_value(), self.scale, np.linalg.norm(grad), self.sigma, B), 
        # '\n\tdelta_t={} '.format(update), 
        # '\n\tM={} \n\tw={}'.format(self.M, np.array(self.ws)[:self.h]))

        # play u_{t + 1}
        self.t += 1
        self.ws.appendleft(w)
        u = self._compute_u()
        self.prev_value = self.get_value()
        self.set_value(u)        
        pass

    def _compute_u(self):
        u = self.M[-1]
        if self.method == 'FKM':
            self.ns = np.random.randn(self.h + 1) * self.sigma
            for j in range(min(self.h, len(self.ws))):
                M, n, w = self.M[j], self.ns[j], self.ws[j]
                u += ((1 - self.sigma) * M + n) * w
                # u += (M + n) * w
            u += self.ns[-1]  # add bias noise
        elif self.method == 'REINFORCE':
            n = np.random.randn() * self.sigma
            for j in range(min(self.h, len(self.ws))):
                M, w = self.M[j], self.ws[j]
                u += M * w
            u += n
            self.ns.appendleft(n)
        if self.B is not None:  # add what is needed for system identification
            self.r = np.random.randn() * self.sigma
            u += self.r
        return u

    def _compute_update(self, grad):
        t = self.t
        # given array of gradients, compute ADAM update
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        alpha_t = self.meta_lr * np.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t)
        update = alpha_t * self.m / (self.v ** 0.5 + self.eps)
        return update
    