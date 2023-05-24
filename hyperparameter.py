import numpy as np
from collections import deque

from wrappers import FloatWrapper

"""
PIPELINE SHOULD BE:
    1. make object with something like `lr = FloatHyperparameter(h=3, initial_value=0.54)`
    2. use it as a float in ways like `opt = optim.Adam(lr)`
    2. every so often, call `lr.step(o=current val error, p=optional prediction)`
    
    ((if prediction is None we use the last `o` as our prediction))
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
        self.M = np.zeros(self.h)  # self.M[i - 1] = M_i^t
        self.ns = deque(maxlen=self.h)  # self.ns[i] = n_{t-i}
        self.disturbances = deque(maxlen=2 * self.h)  # self.disturbances[-i] = epsilon_{t-i}
        self.prev_o = None  # o_{t-1}, which we set p_t to if predictions aren't provided
        self.scale = 0.
        self.t = 0
        
        self.initial = initial_value
    
    def step(self, o: float, prediction: float=None):
        """
        steps the hyperparameter once.
        FIRST STEP DOES NOTHING EXCEPT CACHE PREV OBSERVATION

        Parameters
        ----------
        o : float
            o_t
        prediction : float, optional
            p_t, if left as `None` will result in p_t=o_{t-1}
        """
        
        if self.prev_o is None:  # do nothing first time, so that next time we get accurate scale initialization
            self.prev_o = o
            self.scale = max(self.scale, abs(o))
            return
        
        # observe
        epsilon = o - self.prev_o if prediction is None else o - prediction  # observe new objective and prediction
        self.scale = max(self.scale, abs(o)) # set scale
        
        # REINFORCE update
        for j in range(self.h):
            grad_j = 0.
            for i in range(1, self.h + 1):
                if i + j > len(self.disturbances):
                    break
                grad_j += self.ns[-i] * self.disturbances[-(i + j)]
            grad_j *= o / self.scale
            
            self.M[j] = self.M[j] - grad_j / np.sqrt(self.t + 1)

        # play eta_{t + 1}
        self.t += 1
        self.disturbances.append(epsilon)
        
        n = np.random.randn() * np.sqrt(self.scale)  # n_{t+1}
        eta = 0.
        for i in range(1, self.h + 1):
            if i > len(self.disturbances):
                break
            eta += self.M[i - 1] * self.disturbances[-i]
        eta += n
            
        print('Updated from {} to {}!'.format(self, eta))
        self.set_value(eta)        
        self.ns.append(n)
        pass
        
if __name__ == '__main__':
    np.random.seed(0)
    lr = FloatHyperparameter(5, 0.1)
    
    print(lr, lr + 5, 8 - 2 * lr)
    lr.step(0.00001)
    print(lr, lr + 5, 8 - 2 * lr)
