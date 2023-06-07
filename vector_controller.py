from abc import abstractmethod
import numpy as np
from numpy import ndarray
from typing import Tuple
from numpy._typing import NDArray
from collections import deque

from rescalers import ADAM, D_ADAM, EMA_RESCALE, DoWG

class _Predictor:
    @abstractmethod
    def forward(self,
                ws: ndarray,
                scale: float) -> ndarray:
        """
        Predicts control using the predictor's parameters, suitably perturbed with the given scale.

        Parameters
        ----------
        ws : ndarray
            `(h,)` array of the past disturbances, i.e. `ws = (w_{t}, w_{t-1}, ..., w_{t-h})`
        scale : float
            std dev of noise
        """
        pass
    
    @abstractmethod
    def update(self,
               grad: float,
               ws: ndarray) -> None:
        """
        Updates the predictor's parameters one step

        Parameters
        ----------
        grad : float
            `d_u * (s_t - s_{t-1}) / sigma_{t-1}`
        ws : ndarray
            `(2h,)` array of the past disturbances, i.e. `ws = (w_{t}, w_{t-1}, ..., w_{t-2h})`
        """
        pass

class _FKMLinear(_Predictor):
    def __init__(self,
                 h: int,
                 du: int,
                 grad_rescaler):
        self.h = h
        self.du = du
        self.grad_rescaler = grad_rescaler
        
        self.M = np.zeros((self.h + 1, self.du))  # M[-1] is bias and M[0:h] are meant to multiply disturbances
        self.ns = deque(np.zeros((self.h, self.h + 1, self.du)), maxlen=h)  # leftmost entry is most recent
        pass
    
    def forward(self, 
                ws: ndarray,
                scale: float):
        assert ws.shape == (self.h,)
        
        # generate noise on the sphere
        n = np.random.randn(self.h + 1, self.du)
        n /= np.linalg.norm(n, axis=1)
        self.ns.appendleft(n)
        
        # perturb parameters with scaled noise
        M = (1 - scale) * self.M + scale * n
        u = M[-1] + M[:-1].T @ ws
        return u
    
    def update(self,
               grad: float,
               ws: ndarray):
        grad = grad * np.sum(np.asarray(self.ns), axis=0)
        grad = self.grad_rescaler.step(grad)
        self.M -= grad
        pass

f = _FKMLinear(5, 2, ADAM())
f.update(2, None)
exit(0)

class VectorController(ndarray):
    def __new__(cls, 
                h: int,
                initial_value: NDArray, 
                bounds: Tuple[NDArray, NDArray],
                w_clip_size: float=float('inf'), 
                M_clip_size: float=float('inf'), 
                method='FKM'):
        
        obj = np.asarray(initial_value).view(cls)
        obj.h = h
        obj.initial_value = initial_value
        obj.bounds = bounds
        obj.w_clip_size = w_clip_size
        obj.M_clip_size = M_clip_size
        obj.method = method
        return obj

    # TODO eventually define this to ensure safety when creating wrappers the other ways
    def __array_finalize__(self, obj):
        if obj is None: return
        self.h = getattr(obj, 'h', None)
        self.initial_value = getattr(obj, 'initial_value', None)
        self.bounds = getattr(obj, 'bounds', None)
        self.w_clip_size = getattr(obj, 'w_clip_size', float('inf'))
        self.M_clip_size = getattr(obj, 'M_clip_size', float('inf'))
        self.method = getattr(obj, 'method', None)
        
        # parameters of the controller
        self.M = np.zeros(self.h + 1, self.shape[0])  # self.M[i] = M_i^t, but self.M[-1] is the bias
        self.M[-1] = self.initial_value

        # dynamic parameters of the optimization
        self.update = ADAM(alpha=0.01, betas=(0.9, 0.999))
        # self.update = D_ADAM(betas=(0.9, 0.999), growth_rate=1.02, use_bias_correction=False)
        # self.update = DoWG(alpha=1e4)
        self.w = ADAM(betas=(0.9, 0.999))
        
        self.prev_obj = None  # previous cost f(x_t)
        self.t = 0  # timestep t
        self.B = None  # keep track of system estimates B_t, if needed
        self.r = None  # random variable to estimate system parameter B, if needed
        self.scale = None  # keep track of current scale G_t
        self.sigma = None
        if self.method == 'FKM':
            self.ns = np.zeros(self.h + 1, self.shape[0])  # self.ns[i] = n_{t, i}, but self.ns[-1] = n_{t, 0} is the bias noise
            self.ws = deque([0 for _ in range(self.h)], maxlen=self.h)  # self.ws[i] = epsilon_{t-i-1}
        elif self.method == 'REINFORCE':
            self.ns = deque([0 for _ in range(self.h + 1)], maxlen=self.h + 1)  # self.ns[i] = n_{t-i}
            self.ws = deque([0 for _ in range(2 * self.h)], maxlen=2 * self.h)  # self.ws[i] = epsilon_{t-i-1}
        else:
            raise NotImplementedError()
    
    
        
    
    def step(self, val):
        self += val
        pass
    
# v = np.array([0, 1, 1, 0]).view(VectorWrapper)
v = VectorWrapper([0, 1, 1, 0])
print(v)
print(v.__class__)
print(dir(v))
v.step(1)
print(v)
print(v.__class__)
print(dir(v))