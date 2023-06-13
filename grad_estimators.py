from abc import abstractmethod
from collections import deque

import numpy as np
from numpy import ndarray

from rescalers import Rescaler

class GradientEstimator:
    def __init__(self,
                 h: int,
                 du: int,
                 grad_rescaler: Rescaler):
        """
        Object to predict controls from disturbances and estmate cost gradients w.r.t the parameters.

        Parameters
        ----------
        h : int
            number of past disturbances to predict with
        du : int
            dimension of control
        grad_rescaler : Rescaler
            a rescaler to allow for dynamic or adaptive gradient updates
        """
        self.h = h
        self.du = du
        self.grad_rescaler = grad_rescaler
        
        self.prev_cost = None
        pass
    
    @abstractmethod
    def forward(self,
                ws: ndarray[float],
                sigma: float) -> ndarray:
        """
        Predicts control using the predictor's parameters, suitably perturbed with the given scale.

        Parameters
        ----------
        ws : ndarray
            `(h,)` array of the past disturbances, i.e. `ws = (w_{t}, w_{t-1}, ..., w_{t-h})`
        sigma : float
            std dev of noise
        """
        pass
    
    @abstractmethod
    def update(self,
               grad_hat: float,
               ws: ndarray[float],
               grad_true: ndarray[float]=None) -> None:
        """
        Updates the predictor's parameters one step

        Parameters
        ----------
        grad_hat : float
            `d_u * (s_t - s_{t-1}) / sigma_{t-1}`
        ws : ndarray
            `(2h,)` array of the past disturbances, i.e. `ws = (w_{t}, w_{t-1}, ..., w_{t-2h})`
        grad_true : ndarray, optional
            the actual gradient vector `\nabla_u f`, which should be used to calculate updates if provided, default is None
        """
        pass

class FKMLinear(GradientEstimator):
    def __init__(self,
                 h: int,
                 du: int,
                 grad_rescaler: Rescaler):
        super().__init__(h, du, grad_rescaler)
        self.M = np.zeros((self.h + 1, self.du))  # M[-1] is bias and M[0:h] are meant to multiply disturbances
        self.ns = deque(np.zeros((self.h, self.h + 1, self.du)), maxlen=h)  # leftmost entry is most recent
        pass
    
    def forward(self, 
                ws: ndarray[float],
                sigma: float):
        assert ws.shape == (self.h,)
        
        # generate noise on the sphere
        n = np.random.randn(self.h + 1, self.du)
        n /= np.linalg.norm(n.reshape(-1))
        self.ns.appendleft(n)
        
        # perturb parameters with scaled noise
        M = (1 - sigma) * self.M + sigma * n
        u = M[-1] + M[:-1].T @ ws
        return u
    
    def update(self,
               grad_hat: float,
               ws: ndarray[float],
               grad_true: ndarray[float]=None):
        assert ws.shape == (2 * self.h,)
        
        # compute inner gradients
        if grad_true is not None:
            grad = np.zeros_like(self.M)
            grad[:-1] = grad_true.reshape(-1, 1) @ ws[:self.h].reshape(1, -1)
            grad[-1] = grad_true
        else:
            grad = sum(self.ns)
            grad *= grad_hat
            
        # rescale them, and update parameters
        grad = self.grad_rescaler.step(grad)
        self.M -= grad
        pass

class REINFORCELinear(GradientEstimator):
    def __init__(self,
                 h: int,
                 du: int,
                 grad_rescaler: Rescaler):
        super().__init__(h, du, grad_rescaler)
        self.M = np.zeros((self.h + 1, self.du))  # M[-1] is bias and M[0:h] are meant to multiply disturbances
        self.ns = deque(np.zeros((self.h + 1, self.du)), maxlen=h + 1)  # leftmost entry is most recent
        pass
    
    def forward(self, 
                ws: ndarray[float],
                sigma: float):
        assert ws.shape == (self.h,)
        
        # generate noise on the sphere
        n = np.random.randn(self.du)
        n /= np.linalg.norm(n)
        self.ns.appendleft(n)
        
        # perturb parameters with scaled noise
        u = self.M[-1] + self.M[:-1].T @ ws + sigma * n
        return u
    
    def update(self,
               grad_hat: float,
               ws: ndarray[float],
               grad_true: ndarray[float]=None):
        assert ws.shape == (2 * self.h,)
        
        # compute inner gradients
        if grad_true is not None:
            grad = np.zeros_like(self.M)
            grad[:-1] = grad_true.reshape(-1, 1) @ ws[:self.h].reshape(1, -1)
            grad[-1] = grad_true
        else:
            grad = np.zeros_like(self.M)
            for i in range(1, self.h + 1):
                grad[:-1] += ws[i:i + self.h].reshape(-1, 1) @ list(self.ns)[i].reshape(1, -1)
            grad[-1] = self.ns[0]
            grad *= grad_hat
        
       # rescale them, and update parameters
        grad = self.grad_rescaler.step(grad)
        self.M -= grad
        pass
    
from rescalers import ADAM
p = REINFORCELinear(3, 2, ADAM())
for _ in range(3):
    p.update(2, np.array([1, 2, 1, 4, 1, 6]))