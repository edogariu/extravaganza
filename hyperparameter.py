from typing import Tuple
from collections import deque
import numpy as np

from wrappers import FloatWrapper

TRANSFORMATION = 'w_sigmoid'  # ['sigmoid', 'rescale', 'w_sigmoid', 'w_rescale']

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
                 grad_clip_size: float=float('inf'),
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
        grad_clip_size : float
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
        self.grad_clip_size = grad_clip_size
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
        self.M = np.zeros(self.h + 1)  # self.M[i] = M_i^t, but self.M[-1] is the bias
        self.M[-1] = self.value[0]
        self.D = 1  # D_t, distance in M-space traveled during the optimization 
        self.prev = None  # previous cost f(x_t)
        self.t = 1  # timestep t
        self.B = None  # keep track of system estimates B_t, if needed
        self.r = None  # random variable to estimate system parameter B, if needed
        self.scale = None  # keep track of current scale G_t
        self.G0 = None  # keep track of initial scale G_0
        if method == 'FKM':
            self.ns = np.zeros(self.h + 1)  # self.ns[i] = n_{t, i}, but self.ns[-1] = n_{t, 0} is the bias noise
            self.ws = deque(maxlen=self.h)  # self.ws[i] = epsilon_{t-i-1}
        elif method == 'REINFORCE':
            self.ns = deque(maxlen=self.h)  # self.ns[i] = n_{t-i}
            self.ws = deque(maxlen=2 * self.h)  # self.ws[i] = epsilon_{t-i-1}
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
        if self.prev is None:
            if obj == 0: return  
            self.prev = obj
            self.scale = self.G0 = abs(obj)
            self.sigma = min(0.1, self.G0)
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
        pred = self.prev + self.get_value() * B  # \hat{s_t} = f(x_{t-1}) + u_{t-1} * B
        w = obj - pred - self.quadratic_term * self.get_value() ** 2  # w_t = s_t - \hat{s_t} (maybe with - u_{t-1}^2)
        self.prev = (1 - 0.1) * obj
               
        # set scale
        self.sigma = min(0, self.G0 / (self.t ** 0.25))
        # self.scale = max(self.scale, abs(obj))
        scale_lr = 0.01# 1 / (self.t ** 0.5)
        self.scale = (1 - scale_lr) * self.scale + scale_lr * obj
        self.D = max(self.D, np.linalg.norm(self.M))
        meta_lr = self.D / (self.scale * self.t ** 0.5)
                
        # update
        if grad_u is not None:
            grad = grad_u * np.array(self.ws)[:self.h]
            grad_bias = grad_u
        else:
            if self.method == 'FKM':
                grad = self.ns[:-1] * obj / self.sigma
                grad_bias = self.ns[-1] * obj / self.sigma
            elif self.method == 'REINFORCE':
                grad = np.array(self.ns)[1:] * np.array(self.ws)[:len(self.ns) - 1] * obj / self.sigma
                grad_bias = self.ns[0] * obj / self.sigma if len(self.ns) > 0 else 0
                
        if self.rescale_u:
            d = d_rescale(self.value[0], self.interval)
            grad *= d
            grad_bias *= d
        elif self.rescale_w:
            d = d_rescale_11(self.value[0], self.interval)
            grad *= d
            grad_bias *= d
            
        print('u_t={:.4f} \tG_t={:.4f} \tD_t={:.4f}'.format(self.get_value(), self.scale, self.D), 
        '\nsigma_t={:.4f} \teta_t={} \t||grad||={:.4f}'.format(self.sigma, meta_lr, np.linalg.norm([*grad, grad_bias])), 
        '\tB={:.4f} \n\tM={} \n\tw={}'.format(B, self.M.round(3), np.array(self.ws)[:self.h].round(3)))
        
        grad = np.clip(grad, -self.grad_clip_size, self.grad_clip_size)
        grad_bias = np.clip(grad_bias, -self.grad_clip_size, self.grad_clip_size)
        # grad /= np.maximum(1, np.abs(grad)); grad_bias /= max(1, np.abs(grad_bias))
        self.M[:len(grad)] -= meta_lr * grad
        self.M[-1] -= meta_lr * grad_bias
        
        # clip some more
        w = np.clip(w, -self.w_clip_size, self.w_clip_size)
        if self.rescale_w: 
            w_scale = self.G0
            v = 0.5
            w = rescale(w, [-w_scale, w_scale]) / w_scale  # w \in [-1, 1]
            self.M[:-1] = np.clip(self.M[:-1], -v / self.h, v / self.h)  # M \cdot w \in [-1, 1]
            self.M[-1] = np.clip(self.M[-1], -(1 - v), 1 - v)
        else:
            w = np.clip(w, -self.G0, self.G0)
            self.M[:-1] = np.clip(self.M[:-1], -self.M_clip_size, self.M_clip_size)

        # play u_{t + 1}
        self.t += 1
        self.ws.appendleft(w)
        u = self._compute_u()
        self.set_value(u)        
        pass

    def _compute_u(self):
        u = self.M[-1] #+ self.initial_value
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
    