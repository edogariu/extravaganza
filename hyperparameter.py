# WHERE THE `Hyperparameter` class is defined

from typing import Callable, Any
import numpy as np
from collections import deque

from utils import FloatWrapper

"""
PIPELINE SHOULD BE:
    1. make object with something like `lr = Hyperparameter(h=3, update_fn=lambda v: update_SGD_lr(v), o_0=0.54)`
    2. every so often, call `lr.step(o=current val error, p=optional prediction)`
    
    ((if prediction is None we use the last o))
"""

class FloatHyperparameter(FloatWrapper):
    def __init__(self, h: int, initial_value: float):
        """
        GPC for a float hyperparameter. 
        Initializes hyperparameter values based on initial observation `(o_0, p_0)`

        Parameters
        ----------
        h : float
            window size
        initial_value : float, optional
            initial value for the hyperparameter to take
        """
        self.h = h
        super().__init__(initial_value)
        
        # dynamic parameters of the optimization
        self.M = np.zeros(self.h)
        self.ns = deque([0.], maxlen=self.h)
        self.epsilons = deque([0.], maxlen=self.h)
        self.prev_o = 0.
        self.scale = 0.
        self.t = 0
    
    def step(self, o: float, prediction: float=None):
        """
        steps the hyperparameter once

        Parameters
        ----------
        o : float
            o_t
        prediction : float, optional
            p_t, if left as `None` will result in p_t=o_{t-1}
        """
        assert self.M is not None, 'must call initialize() first'
        
        # observe
        epsilon = o - self.prev_o if prediction is None else o - prediction  # observe new objective and prediction
        self.scale = max(self.scale, o)  # set scale
        
        # REINFORCE update
        # val = sum([n * epsilon for n, epsilon in zip(self.ns, self.epsilons)])  # TODO check if this should use epsilon_{t-1} or epsilon_{t-i}
        val = sum([n * self.epsilons[-1] for n in self.ns])
        self.M -= self.scale * o * val / np.sqrt(self.t + 1)  # TODO use better scalings
        
        # play eta_{t + 1}
        self.t += 1
        self.epsilons.append(epsilon)
        n = np.random.randn() * np.sqrt(self.scale)  # n_t
        self.ns.append(n)
        eta = n
        for m, epsilon in zip(self.M, self.epsilons):  # m is M_i^t and epsilon is \hat{epsilon}_{t-i}
            eta += m * epsilon
        print('Updated learning rate from {} to {}!'.format(self, eta))
        self.set_value(eta)        
        pass
        
if __name__ == '__main__':
    np.random.seed(0)
    lr = FloatHyperparameter(5, 0.1)
    
    print(lr, lr + 5, 8 - 2 * lr)
    lr.step(0.00001)
    print(lr, lr + 5, 8 - 2 * lr)



# # WHERE THE `Hyperparameter` class is defined

# from typing import Callable, Any
# import numpy as np
# from collections import deque

# from utils import FloatWrapper

# """
# PIPELINE SHOULD BE:
#     1. make object with something like `lr = Hyperparameter(h=3, update_fn=lambda v: update_SGD_lr(v), o_0=0.54)`
#     2. every so often, call `lr.step(o=current val error, p=optional prediction)`
    
#     ((if prediction is None we use the last o))
# """

# class FloatHyperparameter(FloatWrapper):
#     def __init__(self, h: int, o_0: float, p_0 : float=None):
#         """
#         GPC for a float hyperparameter. 
#         Initializes hyperparameter values based on initial observation `(o_0, p_0)`

#         Parameters
#         ----------
#         o : float
#             o_0
#         prediction : float, optional
#             p_0, if left `None` will be used as p_0=0
#         """
#         self.value = FloatWrapper(0.)
        
#         self.h = h
        
#         # dynamic parameters of the optimization
#         epsilon = o_0 if p_0 is None else o_0 - p_0
#         self.M = np.zeros(self.h)
#         self.ns = deque([0.], maxlen=self.h)
#         self.epsilons = deque(maxlen=self.h)
#         self.prev_o = o_0
#         self.scale = o_0
        
#         # play eta_1
#         self.t = 1
#         self.epsilons.append(epsilon)
#         eta = self._compute_eta()
#         super().__init__(eta)
#         pass
    
#     def step(self, o: float, prediction: float=None):
#         """
#         steps the hyperparameter once

#         Parameters
#         ----------
#         o : float
#             o_t
#         prediction : float, optional
#             p_t, if left as `None` will result in p_t=o_{t-1}
#         """
#         assert self.M is not None, 'must call initialize() first'
        
#         # observe
#         epsilon = o - self.prev_o if prediction is None else o - prediction  # observe new objective and prediction
#         self.scale = max(self.scale, o)  # set scale
        
#         # REINFORCE update
#         # val = sum([n * epsilon for n, epsilon in zip(self.ns, self.epsilons)])  # TODO check if this should use epsilon_{t-1} or epsilon_{t-i}
#         val = sum([n * self.epsilons[-1] for n in self.ns])
#         self.M -= self.scale * o * val / np.sqrt(self.t + 1)  # TODO use better scalings
        
#         # play eta_{t + 1}
#         self.t += 1
#         self.epsilons.append(epsilon)
#         self.set_value(self._compute_eta())
#         pass
        
#     def _compute_eta(self):
#         n = np.random.randn() * np.sqrt(self.scale)  # n_t
#         self.ns.append(n)
#         eta = n
#         for m, epsilon in zip(self.M, self.epsilons):  # m is M_i^t and epsilon is \hat{epsilon}_{t-i}
#             eta += m * epsilon
#         return eta
        
# if __name__ == '__main__':
#     np.random.seed(0)
#     lr = FloatHyperparameter(5, None, 200)
    
#     print(lr, lr + 5, 8 - 2 * lr)
#     lr.step(0.00001)
#     print(lr, lr + 5, 8 - 2 * lr)
