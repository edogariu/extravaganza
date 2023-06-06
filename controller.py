from typing import Tuple
from collections import deque
import numpy as np

from wrappers import FloatWrapper
from utils import EMA_RESCALE, ADAM, D_ADAM, DoWG, sigmoid, d_sigmoid, inv_sigmoid

use_sigmoid = True

def rescale(t, bounds):
    """
    rescales from `[0, 1] -> [tmin, tmax]`
    """
    tmin, tmax = bounds
    if use_sigmoid: t = sigmoid(t)
    return tmin + (tmax - tmin) * t

def d_rescale(t, bounds):
    tmin, tmax = bounds
    d = tmax - tmin
    if use_sigmoid: d *= d_sigmoid(t)
    return d

def inv_rescale(s, bounds):
    tmin, tmax = bounds
    t = (s - tmin) / (tmax - tmin)
    if use_sigmoid: t = inv_sigmoid(t)
    return t

class FloatController(FloatWrapper):
    def __init__(self, 
                 h: int, 
                 initial_value: float, 
                 bounds: Tuple[float, float]=(0, 1),
                 w_clip_size: float=float('inf'), 
                 M_clip_size: float=float('inf'), 
                 B_clip_size: float=float('inf'),
                 cost_clip_size: float=float('inf'), 
                 update_clip_size: float=float('inf'),
                 method='FKM'):
        """
        GPC for a float hyperparameter. 

        Parameters
        ----------
        h : int
        initial_value : float
        bounds : Tuple[float, float]
        w_clip_size : float
        M_clip_size : float
        B_clip_size : float
        cost_clip_size : float
        update_clip_size : float
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
        self.method = method
        
        self.bounds = bounds
        self.umin, self.umax = bounds
        super().__init__(inv_rescale(initial_value, self.bounds))
        
        # dynamic parameters of the optimization
        # for meta lr updates
        # self.update = ADAM(alpha=0.01, betas=(0.9, 0.999))
        self.update = D_ADAM(betas=(0.9, 0.999), growth_rate=1.02, use_bias_correction=True)
        # self.update = DoWG(alpha=1e5)
        # self.obj = ADAM(betas=(0.9, 0.999))
        # self.obj = EMA_RESCALE(beta=0)
        
        self.M = np.zeros(self.h + 1)  # self.M[i] = M_i^t, but self.M[-1] is the bias
        self.M[-1] = self.value[0]
        self.prev_obj = None  # previous cost f(x_t)
        self.prev_control = self.get_value()  # previous value u_{t-1}
        self.t = 0  # timestep t
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
        return rescale(super().get_value(), self.bounds)
    
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
        
        # obj = self.obj.step(obj)
        
        # cache observation on first step
        if self.t == 0:
            if obj == 0: return  
            self.prev_obj = obj
            self.scale = abs(obj)
            self.sigma = min(1e-5, self.scale)
            if B is None:
                self.r = np.random.randn() * self.sigma
                self.B = self.r * obj
            self.t += 1
            return        
        
        # observe
        if B is None:
            b_lr = 1 / self.t
            self.B = (1 - b_lr) * self.B + b_lr * obj * self.r / self.sigma
            self.B = np.clip(self.B, -self.B_clip_size, self.B_clip_size)
            B = self.B
        # pred = self.prev_obj + self.prev_control * B  # \hat{s_t} = f(x_{t-1}) + B
        pred = self.prev_control
        w = obj - pred  # w_t = s_t - \hat{s_t}
        w = np.clip(w, -self.w_clip_size, self.w_clip_size)
        self.prev_obj = obj
                
        # compute scale
        # scale_lr = 0.01# 1 / (self.t ** 0.5)
        # self.scale = (1 - scale_lr) * self.scale + scale_lr * abs(obj)
        self.scale = max(self.scale, abs(obj))
        self.sigma = min(0.01, self.scale / (self.t ** 0.25))
                
        # calc gradients and update
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
                
        grad *= d_rescale(float(self.value[0]), self.bounds)
        update = self.update.step(grad, iterate=self.M)
        
        if self.t < self.h:  # only observe for `h` steps
            self.t += 1
            self.ws.appendleft(w)
            return
        
        update = np.clip(update, -self.update_clip_size, self.update_clip_size)
        self.M -= update
                
        # clip some more
        # w = np.clip(w, -1, 1)
        # w = self.w.step(np.clip(w, -self.w_clip_size, self.w_clip_size))
        # self.M[:-1] = np.clip(self.M[:-1], -1 / self.h, 1 / self.h)
        # self.M[-1] = np.clip(self.M[-1], inv_rescale(self.umin, self.bounds), inv_rescale(self.umax, self.bounds))
        
        print('u_t={:.4f} \tG_t={:.4f} \t||grad||={:.4f} \tsigma_t={:.4f}\tB={:.4f}'.format(self.get_value(), self.scale, np.linalg.norm(grad), self.sigma, B), 
        '\n\tdelta_t={} '.format(update), 
        '\n\tM={} \n\tw={}'.format(self.M, np.array(self.ws)[:self.h]))

        # play u_{t + 1}
        self.t += 1
        self.ws.appendleft(w)
        u = self._compute_u()
        self.prev_control = self.get_value()
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
