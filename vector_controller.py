import numpy as np
from numpy import ndarray, ufunc
from typing import Any, Literal as L, Tuple
from collections import deque

from rescalers import ADAM, D_ADAM, EMA_RESCALE, DoWG
from grad_estimators import FKMLinear, REINFORCELinear
from utils import rescale, d_rescale, inv_rescale

RESCALE = True
USE_SIGMOID = True

ESTIMATORS = {'FKM': FKMLinear,
              'REINFORCE': REINFORCELinear}

class VectorController(ndarray):
    def __new__(cls, 
                h: int,
                initial_value: ndarray, 
                bounds: Tuple[ndarray[float], ndarray[float]]=None,
                delta: float=0.01,
                sigma_max: float=0.1,
                wait_time: int=0,
                w_clip_size: float=float('inf'), 
                M_clip_size: float=float('inf'), 
                method='FKM'):
        assert method in ESTIMATORS, 'invalid method {}, must be one of {}'.format(method, ESTIMATORS.keys())
        
        # args
        obj = np.asarray(initial_value).view(cls)
        obj.h = h
        obj.initial_value = initial_value
        obj.bounds = bounds
        obj.delta = delta
        obj.sigma_max = sigma_max
        obj.wait_time = wait_time
        obj.w_clip_size = w_clip_size
        obj.M_clip_size = M_clip_size
        obj.method = method
        
        # dynamic things 
        obj.state = {
            'ws': deque(np.zeros(2 * obj.h), maxlen=2 * obj.h),  # past 2h disturbances (most recent on left)
            'prev_cost': None,  # previous cost f(x_{t-1})
            'G': None,  # keep track of scale G_t
            't': 0,  # timestep t
            'B': np.zeros_like(obj),  # keep track of system information B_t
            'r': np.zeros_like(obj),  # random variable to estimate system parameter B, if needed
        }
        grad_rescaler = ADAM(alpha=0.01, betas=(0.9, 0.999))
        # grad_rescaler = D_ADAM(betas=(0.9, 0.999), growth_rate=1.02, use_bias_correction=False)
        # grad_rescaler = DoWG(alpha=1e4)
        obj.estimator = ESTIMATORS[obj.method](obj.h, obj.du, grad_rescaler)
        obj.w_rescaler = ADAM(betas=(0.9, 0.999))
        return obj

    # TODO eventually define this to ensure safety when creating wrappers the other ways
    def __array_finalize__(self, obj):
        if obj is None: return
        self.h = getattr(obj, 'h', None)
        self.du = obj.shape[0]
        self.initial_value = getattr(obj, 'initial_value', None)
        self.bounds = getattr(obj, 'bounds', None)
        self.delta = getattr(obj, 'delta', None)
        self.sigma_max = getattr(obj, 'sigma_max', None)
        self.wait_time = getattr(obj, 'wait_time', None)
        self.w_clip_size = getattr(obj, 'w_clip_size', None)
        self.M_clip_size = getattr(obj, 'M_clip_size', None)
        self.method = getattr(obj, 'method', None)
        self.state = getattr(obj, 'state', {})
        self.estimator = getattr(obj, 'estimator', None)
        self.w_rescaler = getattr(obj, 'w_rescaler', None)
        pass
    
    def __array_ufunc__(self, ufunc: ufunc, method, *inputs: Any, **kwargs: Any) -> Any:
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, VectorController):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = (None,) * ufunc.nout    
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(VectorController)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results
        
    def step(self, 
             cost: float, 
             B_true: ndarray=None, 
             grad_true: ndarray=None):
        """
        steps the controller once.
        FIRST STEP DOES NOTHING EXCEPT CACHE PREV OBSERVATION

        Parameters
        ----------
        cost : float
            `f(x_t)`
        B_true : ndarray, optional
            `B` param of the LDS, default is None
        grad_true : ndarray, optional
            gradient of cost w.r.t. control variable `u_t`, default is None
        """
        prev_cost, G, t, B, r, ws = self.state['prev_cost'], self.state['G'], self.state['t'], self.state['B'], self.state['r'], self.state['ws']
        arr_ws = np.asarray(ws)
        
        if t <= self.wait_time:  # observe and do nothing
            G = abs(cost)
            if B_true is not None: B = B_true
        else:
            # observe
            diff = cost - prev_cost
            sigma = min(self.sigma_max, G / t ** 0.25)
            
            # sys id
            if B_true is None:  # update system estimate and draw noise for next estimate if required
                B_momentum = 1 - 1 / t
                B = B_momentum * B + (1 - B_momentum) * diff * r / sigma
                r = np.random.randn(self.du)
            else:
                B = B_true
            
            pred_cost = prev_cost + np.dot(B, self)
            w = cost - pred_cost
            w = np.clip(w, -self.w_clip_size, self.w_clip_size)
            w = self.w_rescaler.step(w)
            ws.appendleft(w)
            
            # predict the control `u`
            u = self.estimator.forward(arr_ws[:self.h], sigma) + r * sigma
            self[:] = u
            
            # update the estimator's parameters
            grad_hat = self.du * diff / sigma
            if self.bounds is not None: # rescale if necessary
                
            self.estimator.update(grad_hat, arr_ws, grad_true=grad_true)
            
        prev_cost = cost
        t += 1
        self.state['prev_cost'], self.state['G'], self.state['t'], self.state['B'], self.state['r'], self.state['ws'] = prev_cost, G, t, B, r, ws
        pass
            
    
v = VectorController(h=5, initial_value=np.random.randn(5), bounds=None, wait_time=0)
# print(v)
from time import perf_counter; s = perf_counter()
for _ in range(1000): v.step(1)
print((perf_counter() - s) / 1000, ' seconds per step')
# print(v)
# print(np.exp(v))
# print(v)