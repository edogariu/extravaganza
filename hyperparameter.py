from typing import Tuple
from collections import deque
import numpy as np

from wrappers import FloatWrapper

class FloatHyperparameter(FloatWrapper):
    def __init__(self, 
                 h: int, 
                 initial_value: float, 
                 initial_scale: float, 
                 interval: Tuple[float, float]=(-float('inf'), float('inf')),
                 w_clip_size: float=float('inf'), 
                 M_clip_size: float=float('inf'), 
                 B_clip_size: float=float('inf'),
                 cost_clip_size: float=float('inf'), 
                 grad_clip_size: float=float('inf'),
                 quadratic_term: float=0,
                 nonnegative: bool=False,
                 method='FKM'):
        """
        GPC for a float hyperparameter. 

        Parameters
        ----------
        h : float
            window size
        initial_value : float
            initial value for the hyperparameter to take
        initial_scale : float
        interval : Tuple[float, float]
            bounds the hyperparameter can take
        w_clip_size : float
        M_clip_size : float
        B_clip_size : float
        cost_clip_size : float
        grad_clip_size : float
        quadratic_term : float
        nonnegative : bool
        grad_estimation_method : str
        """
        
        assert method in ['FKM', 'REINFORCE']
        
        self.h = h
        self.initial_scale = initial_scale
        self.w_clip_size = w_clip_size
        self.M_clip_size = M_clip_size
        self.B_clip_size = B_clip_size
        self.cost_clip_size = cost_clip_size
        self.grad_clip_size = grad_clip_size
        self.quadratic_term = quadratic_term
        self.nonnegative = nonnegative
        self.method = method
        super().__init__(initial_value if not self.nonnegative else np.sqrt(initial_value))
        self.initial = initial_value
        self.interval = interval if not self.nonnegative else np.sign(interval) * np.sqrt(np.abs(interval))
        
        # dynamic parameters of the optimization
        if method == 'REINFORCE':
            self.M = np.zeros(self.h + 1)  # self.M[i] = M_i^t, but self.M[-1] is the bias
            self.ns = deque(maxlen=self.h)  # self.ns[i] = n_{t-i}
            self.ws = deque(maxlen=2 * self.h)  # self.ws[i] = epsilon_{t-i}
            self.prev = None
            self.t = 1
            self.B = None
            self.r = None
        elif method == 'FKM':
            self.M = np.zeros(self.h + 1)  # self.M[i] = M_i^t, but self.M[-1] is the bias
            self.ns = np.zeros(self.h + 1)  # self.ns[i] = n_{t, i}, but self.ns[-1] = n_{t, 0} is the bias noise
            self.ws = deque(maxlen=self.h)  # self.ws[i] = epsilon_{t-i}
            self.prev = None  # p_{t-1}
            self.t = 1
            self.B = None
            self.r = None
            self.s = 0.1
        else:
            raise NotImplementedError()
    
    def get_value(self) -> float:  # square u_t if we desire nonnegative controls
        return super().get_value() ** 2 if self.nonnegative else super().get_value()
    
    def step(self, obj: float, B: float=None, grad_u: float=None):
        """
        steps the hyperparameter once.
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
        if grad_u is not None: grad_u = np.clip(grad_u, -self.grad_clip_size, self.grad_clip_size)
        
        # cache observation on first step
        if self.prev is None:  
            self.prev = obj
            if B is None:
                self.r = np.random.randn() * self.s
                self.B = self.r * obj
            return
        
        # observe
        if B is None:
            b_lr = 1 / self.t
            self.B = (1 - b_lr) * self.B + b_lr * obj * self.r / self.s
            self.B = np.clip(self.B, -self.B_clip_size, self.B_clip_size)
            B = self.B
        
        # print('obj={}, B={}, M={}'.format(obj, B, self.M))
        
        pred = self.prev + self.get_value() * B  # f(x_t) + u_t * B
        w = obj - pred - self.quadratic_term * self.get_value() ** 2
        self.prev = obj
        
        # clip again
        w = np.clip(w, -self.w_clip_size, self.w_clip_size)
        
        # set scale
        scale = self.initial_scale / (self.t ** 0.25)
        meta_lr = 1 / (self.h * np.sqrt(self.t))
        
        # update
        for j in range(self.h):
            if grad_u is not None:   # use provided gradients
                if j >= len(self.ws): break
                grad_j = grad_u * self.ws[j]
            else:  # estimate the gradients!!!
                if self.method == 'FKM':
                    if j >= len(self.ns): break
                    grad_j = self.ns[j] * obj / scale
                elif self.method == 'REINFORCE':
                    if j + 1 >= len(self.ns): break
                    grad_j = self.ns[j + 1] * self.ws[j] * obj / scale
            
            if self.nonnegative: grad_j *= 2 * super().get_value()        
            # grad_j /= abs(grad_j)
            self.M[j] -= meta_lr * grad_j 
        
        if grad_u is not None:  # update bias
            grad_bias = grad_u
        else:
            if self.method == 'FKM':
                grad_bias = self.ns[-1] * obj / scale
            elif self.method == 'REINFORCE':
                grad_bias = self.ns[0] * obj / scale if len(self.ns) > 0 else 0
        
        if self.nonnegative: grad_bias *= 2 * super().get_value()
        # grad_bias /= abs(grad_bias)
        self.M[-1] -= meta_lr * grad_bias
            
        # clip some more
        self.M = np.clip(self.M, -self.M_clip_size, self.M_clip_size)
        
        # play u_{t + 1}
        self.t += 1
        self.ws.appendleft(w)

        u = self._compute_u(scale)
        u = np.clip(u, *self.interval)
        
        # # print('Updated from {} to {} using {} with disturbances {}!'.format(self, u, self.M, list(self.ws)[:self.h]))
        # old = self.get_value()
        self.set_value(u)        
        # print('Updated from {} to {}!'.format(old, self.get_value()))
        pass

    def _compute_u(self, scale: float):
        u = self.M[-1] # self.initial
        if self.method == 'FKM':
            self.ns = np.random.randn(self.h + 1) * scale
            for j in range(min(self.h, len(self.ws))):
                M, n, w = self.M[j], self.ns[j], self.ws[j]
                u += (M + n) * w
            u += self.ns[-1]  # add bias noise
            
            if self.B is not None:  # add what is needed for system identification
                self.r = np.random.randn() * self.s
                u += self.r
            
        elif self.method == 'REINFORCE':
            n = np.random.randn() * scale
            for j in range(min(self.h, len(self.ws))):
                M, w = self.M[j], self.ws[j]
                u += M * w
            u += n  # add bias noise
            self.ns.appendleft(n)
        return u
    