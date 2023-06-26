from abc import abstractmethod
from typing import Tuple

import numpy as np
from numpy import ndarray

class Rescaler:
    @abstractmethod
    def step(self, 
             val: ndarray,
             iterate: ndarray=None) -> ndarray:
        """
        steps the rescaler, returning a rescaled version of `val`

        Parameters
        ----------
        val : ndarray
            the array to be dynamically rescaled
        iterate : ndarray, optional
            any iterates helpful for the computation, by default None
        """
        pass
    
    def __call__(self, 
                 val: ndarray,
                 iterate: ndarray=None) -> ndarray:
        return self.step(val, iterate)

class IDENTITY(Rescaler):
    def __init__(self) -> None:
        super().__init__()
        
    def step(self, val: ndarray, iterate: ndarray = None) -> ndarray:
        return val

class FIXED_RESCALE(Rescaler):
    def __init__(self,
                 alpha: float=1,
                 beta: float=0):
        self.alpha = alpha
        self.beta = beta
        
        self.m = 0.
        pass
    
    def step(self, 
             val: ndarray, 
             iterate: ndarray=None):
        self.m = self.beta * self.m + (1 - self.beta) * val
        value = self.alpha * self.m
        return value

class EMA_RESCALE(Rescaler):
    def __init__(self,
                 alpha: float=1,
                 beta: float=0.9,
                 eps: float=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
        self.m = 0.
        pass
    
    def step(self, 
             val: ndarray, 
             iterate: ndarray=None):
        self.m = self.beta * self.m + (1 - self.beta) * val
        value = self.alpha * val / (np.abs(self.m) + self.eps)
        return value

class ADAM(Rescaler):
    def __init__(self, 
                 alpha: float=1, 
                 betas: Tuple[float, float]=(0.9, 0.999), 
                 eps: float=1e-8,
                 use_bias_correction: bool=False):
        """
        exponential moving average for first and second moments. `self.value` will w.h.p. have norm <= alpha

        Parameters
        ----------
        alpha : float, optional
            multiplier/upper bound, by default 1
        betas : Tuple[float, float], optional
            momentum parameters for first and second moments, respectively, by default (0.9, 0.999)
        eps : float, optional
            small positive value, by default 1e-8
        use_bias_correction : bool, optional
            whether to compute ADAM bias correction, by default False
        """
        self.alpha = alpha
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.use_bias_correction = use_bias_correction
        
        self.m = 0.; self.v = 0.
        self.t = 1
        pass
    
    def step(self, 
             val: ndarray, 
             iterate: ndarray=None):
        bias_correction = (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t) if self.use_bias_correction else 1
        alpha_t = self.alpha * bias_correction
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * val
        self.v = self.beta2 * self.v + (1 - self.beta2) * val ** 2
        
        value = alpha_t * self.m / ((self.v) ** 0.5 + self.eps)
        self.t += 1
        return value
    
    
class D_ADAM(Rescaler):
    def __init__(self, 
                 alpha: float=1, 
                 betas: Tuple[float, float]=(0.9, 0.999), 
                 eps: float=1e-8,
                 use_bias_correction: bool=False,
                 d0: float=1e-6,
                 growth_rate: float=float('inf')):
        """
        d-adaptive optimizer. 

        Parameters
        ----------
        betas : Tuple[float, float], optional
            momentum parameters for first and second moments, respectively, by default (0.9, 0.999)
        alpha : float, optional
            multiplier/upper bound, by default 1
        eps : float, optional
            small positive value, by default 1e-8
        """
        
        self.alpha = alpha
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.use_bias_correction = use_bias_correction
        self.growth_rate = growth_rate
        
        self.d = d0
        self.m, self.v, self.s, self.r = np.zeros((4, 1))
        self.t = 1
        pass

    def step(self, 
             val: ndarray, 
             iterate: ndarray=None):
        if isinstance(val, float): val = np.array(val)
        bias_correction = (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t) if self.use_bias_correction else 1
        alpha_t = self.d * self.alpha * bias_correction
        sqrt_beta2 = self.beta2 ** 0.5
        denom = self.v ** 0.5 + self.eps
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * val * alpha_t  # NOTE the `* alpha_t` is not in vanilla ADAM
        self.v = self.beta2 * self.v + (1 - self.beta2) * val ** 2
        self.s = sqrt_beta2 * self.s + (1 - sqrt_beta2) * val * alpha_t
        self.r = sqrt_beta2 * self.r + (1 - sqrt_beta2) * np.dot(val.reshape(-1), self.s.reshape(-1) / denom.reshape(-1)) * alpha_t
        d_hat = self.r / ((1 - sqrt_beta2) * np.abs(self.s).sum() + self.eps)
        self.d = max(self.d, min(d_hat, self.d * self.growth_rate))
        
        value = self.m / denom
        self.t += 1
        return value
    
class DoWG(Rescaler):
    def __init__(self,
                 alpha: float=1,
                 d0: float=1e-6,
                 eps: float=1e-8):
        self.alpha = alpha
        self.d = d0
        self.eps = eps
        
        self.m = 0.
        self.x0 = None
        pass

    def step(self, 
             val: ndarray, 
             iterate: ndarray=None):
        assert iterate is not None
        if self.x0 is None: self.x0 = iterate
        
        self.d = max(self.d, np.linalg.norm(iterate - self.x0))
        self.m = self.m + self.d ** 2 * np.dot(val.reshape(-1), val.reshape(-1))
        eta = self.alpha * self.d ** 2 / (self.m ** 0.5 + self.eps)
        value = eta * val
        return value
    