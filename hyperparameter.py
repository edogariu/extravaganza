import numpy as np
from collections import deque

from wrappers import FloatWrapper

# TESTING THINGS

class FloatHyperparameter(FloatWrapper):
    def __init__(self, h: int, initial_value: float, initial_scale: float, clip_size: float):
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
        self.initial_scale = initial_scale
        self.clip_size = clip_size
        super().__init__(initial_value)
        
        # dynamic parameters of the optimization
        self.M = np.zeros(self.h)  # self.M[i - 1] = M_i^t
        self.ns = np.zeros(self.h)  # self.ns[i - 1] = n_{t, i}
        self.disturbances = deque(maxlen=self.h)  # self.disturbances[i] = epsilon_{t-i}
        self.prev_o = None  # o_{t-1}, which we set p_t to if predictions aren't provided
        self.t = 0
        
        self.initial = initial_value
    
    def step(self, o: float, grad_eta: float, prediction: float=None):
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
            return
        
        # observe
        prediction = self.prev_o + sum([m * eps for m, eps in zip(self.M, self.disturbances)])
        epsilon = o - prediction
        self.prev_o = prediction
        epsilon = np.clip(epsilon, -self.clip_size, self.clip_size)
        
        # FKM update
        for j in range(self.h):
            if j >= len(self.disturbances): break
            grad_j = grad_eta * self.disturbances[j]
            self.M[j] -= grad_j / (self.h * np.sqrt(self.t + 1))

        # play eta_{t + 1}
        self.t += 1
        self.disturbances.appendleft(epsilon)
        self.ns = np.random.randn(self.h) * np.sqrt(self.initial_scale / self.t ** 0.25)
        
        eta = self.initial
        for M, n, epsilon in zip(self.M, self.ns, self.disturbances):
            eta += (M + n) * epsilon
        # eta = abs(eta)
            
        print('Updated from {} to {}!'.format(self, eta))
        self.set_value(eta)        
        pass

# # FKM METHOD
# """
# PIPELINE SHOULD BE:
#     1. make object with something like `lr = FloatHyperparameter(h=3, initial_value=0.54)`
#     2. use it as a float in ways like `opt = optim.Adam(lr)`
#     2. every so often, call `lr.step(o=current val error, p=optional prediction)`
    
#     ((if prediction is None we use the last `o` as our prediction))
# """

# class FloatHyperparameter(FloatWrapper):
#     def __init__(self, h: int, initial_value: float):
#         """
#         GPC for a float hyperparameter. 
#         Initializes hyperparameter values based on initial observation `(o_0, p_0)`

#         Parameters
#         ----------
#         h : float
#             window size
#         initial_value : float, optional
#             initial value for the hyperparameter to take
#         """
#         self.h = h
#         super().__init__(initial_value)
        
#         # dynamic parameters of the optimization
#         self.M = np.zeros(self.h)  # self.M[i - 1] = M_i^t
#         self.ns = np.zeros(self.h)  # self.ns[i - 1] = n_{t, i}
#         self.disturbances = deque(maxlen=self.h)  # self.disturbances[-i] = epsilon_{t-i}
#         self.prev_o = None  # o_{t-1}, which we set p_t to if predictions aren't provided
#         self.scale = 1e-8
#         self.t = 0
        
#         self.initial = initial_value
    
#     def step(self, o: float, prediction: float=None):
#         """
#         steps the hyperparameter once.
#         FIRST STEP DOES NOTHING EXCEPT CACHE PREV OBSERVATION

#         Parameters
#         ----------
#         o : float
#             o_t
#         prediction : float, optional
#             p_t, if left as `None` will result in p_t=o_{t-1}
#         """
        
#         if self.prev_o is None:  # do nothing first time, so that next time we get accurate scale initialization
#             self.prev_o = o
#             return
        
#         # observe
#         epsilon = o - self.prev_o if prediction is None else o - prediction  # observe new objective and prediction
#         self.scale = max(self.scale, abs(epsilon)) # set scale
        
#         # FKM update
#         for j in range(self.h):
#             grad_j = self.ns[j] * o / self.scale
#             self.M[j] -= grad_j / np.sqrt(self.t + 1)

#         self.M = np.clip(self.M, -0.1, 0.1)
#         epsilon = np.clip(epsilon, -1, 1)

#         # play eta_{t + 1}
#         self.t += 1
#         self.disturbances.appendleft(epsilon)
#         self.ns = np.random.randn(self.h) * np.sqrt(self.scale)
        
#         eta = 0.
#         for M, n, epsilon in zip(self.M, self.ns, self.disturbances):
#             eta += (M + n) * epsilon
            
#         print('Updated from {} to {}!'.format(self, eta))
#         self.set_value(eta)        
#         pass
        

# REINFORCE METHOD
# """
# PIPELINE SHOULD BE:
#     1. make object with something like `lr = FloatHyperparameter(h=3, initial_value=0.54)`
#     2. use it as a float in ways like `opt = optim.Adam(lr)`
#     2. every so often, call `lr.step(o=current val error, p=optional prediction)`
    
#     ((if prediction is None we use the last `o` as our prediction))
# """

# class FloatHyperparameter(FloatWrapper):
#     def __init__(self, h: int, initial_value: float):
#         """
#         GPC for a float hyperparameter. 
#         Initializes hyperparameter values based on initial observation `(o_0, p_0)`

#         Parameters
#         ----------
#         h : float
#             window size
#         initial_value : float, optional
#             initial value for the hyperparameter to take
#         """
#         self.h = h
#         super().__init__(initial_value)
        
#         # dynamic parameters of the optimization
#         self.M = np.zeros(self.h)  # self.M[i - 1] = M_i^t
#         self.ns = deque(maxlen=self.h)  # self.ns[i] = n_{t-i}
#         self.disturbances = deque(maxlen=2 * self.h)  # self.disturbances[-i] = epsilon_{t-i}
#         self.prev_o = None  # o_{t-1}, which we set p_t to if predictions aren't provided
#         self.scale = 0.
#         self.t = 0
        
#         self.initial = initial_value
    
#     def step(self, o: float, prediction: float=None):
#         """
#         steps the hyperparameter once.
#         FIRST STEP DOES NOTHING EXCEPT CACHE PREV OBSERVATION

#         Parameters
#         ----------
#         o : float
#             o_t
#         prediction : float, optional
#             p_t, if left as `None` will result in p_t=o_{t-1}
#         """
        
#         if self.prev_o is None:  # do nothing first time, so that next time we get accurate scale initialization
#             self.prev_o = o
#             self.scale = max(self.scale, abs(o))
#             return
        
#         # observe
#         epsilon = o - self.prev_o if prediction is None else o - prediction  # observe new objective and prediction
#         self.scale = max(self.scale, abs(o)) # set scale
        
#         # REINFORCE update
#         for j in range(self.h):
#             grad_j = 0.
#             for i in range(1, self.h + 1):
#                 if i + j > len(self.disturbances):
#                     break
#                 # grad_j += self.ns[-i] * self.disturbances[-(i + j)]
                
#             grad_j *= o / self.scale
            
#             self.M[j] = self.M[j] - grad_j / np.sqrt(self.t + 1)

#         # play eta_{t + 1}
#         self.t += 1
#         self.disturbances.append(epsilon)
        
#         n = np.random.randn() * np.sqrt(self.scale)  # n_{t+1}
#         eta = 0.
#         for i in range(1, self.h + 1):
#             if i > len(self.disturbances):
#                 break
#             eta += self.M[i - 1] * self.disturbances[-i]
#         eta += n
            
#         print('Updated from {} to {}!'.format(self, eta))
#         self.set_value(eta)        
#         self.ns.append(n)
#         pass
        
# if __name__ == '__main__':
#     np.random.seed(0)
#     lr = FloatHyperparameter(5, 0.1)
    
#     print(lr, lr + 5, 8 - 2 * lr)
#     lr.step(0.00001)
#     print(lr, lr + 5, 8 - 2 * lr)
