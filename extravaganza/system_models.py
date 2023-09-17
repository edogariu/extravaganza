import logging
from typing import Callable
from copy import deepcopy

import numpy as np

from extravaganza.explorer import Explorer
from extravaganza.controllers import Controller
from extravaganza.lifters import Lifter
from extravaganza.stats import Stats

class Dataset:
    def __init__(self, control_dim: int, obs_dim: int):
        self.control_dim = control_dim
        self.obs_dim = obs_dim
        
        self.os = []  # observations
        self.fs = []  # costs
        self.us = []  # controls
        self.reset_idxs = []  # index of a reset. for example, `self.os[i]` will be the first observation after a reset for every i in `self.reset_idxs`
        pass
    
    def add_transition(self, obs: np.ndarray, cost: float, control: np.ndarray, next_obs: np.ndarray, next_cost: float):
        assert obs.shape == (self.obs_dim,) and control.shape == (self.control_dim,) and next_obs.shape == (self.obs_dim,)
        self.os.append(obs)
        self.fs.append(cost)
        self.us.append(control)
        pass
    
    def reset(self):  # assumes that this will be called before trying to add the reset transition
        self.reset_idxs.append(len(self.os))  # the next transition we see will be a reset
        pass
    
    def get_tail(self, window_size: int):  # returns a `Dataset` object for the most recent `window_size` data points
        l = len(self.os)
        if window_size >= l: return self
        ret = Dataset(self.control_dim, self.obs_dim)
        ret.os = self.os[-window_size:]
        ret.fs = self.fs[-window_size:]
        ret.us = self.us[-window_size:]
        ret.reset_idxs = [i - (l - window_size) for i in self.reset_idxs if i >= (l - window_size)]
        return ret
    
class LiftedController(Controller):
    def __init__(self,
                 controller: Controller,
                 lifter: Lifter):
        # check things make sense
        assert controller.control_dim == lifter.control_dim, 'inconsistent control dims!'
        assert controller.state_dim == lifter.state_dim, 'inconsistent state dims!'
        
        super().__init__()
        self.controller = controller
        self.lifter = lifter
        self.control_dim = controller.control_dim
        self.state_dim = lifter.obs_dim
        self.stats = Stats.concatenate((controller.stats, lifter.stats)) if hasattr(lifter, 'stats') else controller.stats
        self._t = 0
        controller.stats = self.stats
        lifter.stats = self.stats
        self.system_reset_hook = self.controller.system_reset_hook
        pass
    
    def get_control(self, cost: float, obs: np.ndarray) -> np.ndarray:
        assert obs.shape == (self.state_dim,), (obs.shape, self.state_dim)
        state = self.lifter.get_state(obs, cost=cost)
        return self.controller.get_control(cost, state)
    
    def update(self, state: np.ndarray, cost: float, control: np.ndarray, next_state: np.ndarray, next_cost: float):
        # print(jnp.linalg.norm(state[2]) ** 2, cost, '\t\t\t', jnp.linalg.norm(next_state[2]) ** 2, next_cost)  # to confirm we are seeing aligned costs and states
        state, next_state = self.lifter.get_state(state, cost=cost), self.lifter.get_state(next_state, cost=next_cost)
        self._t += 1
        return self.controller.update(state, cost, control, next_state, next_cost)
    
    @property
    def t(self): return self._t
    
    @t.setter
    def t(self, value: int): 
        self._t = value
        self.controller.t = value
        pass
    
    
class OfflineSysid(Controller):   # explore and collect a dataset, then train a lifter, then proceed with that lifted controller
    def __init__(self,
                 control_dim: int, obs_dim: int,
                 explorer: Explorer,
                 make_controller: Callable[[Dataset], Controller],
                 T0: int):        
        super().__init__()
        assert control_dim == explorer.control_dim
        self.control_dim = control_dim
        self.obs_dim = obs_dim
        self.explorer = explorer
        self.make_controller = make_controller
        self.T0 = T0
        
        self.dataset = Dataset(self.control_dim, self.obs_dim)
        self.t = 0
        self.controller = None  # the moment we call get_control() and reach self.t == T0, `self.controller` will no longer be None
        self.stats = None  # same with `self.stats`
        pass

    def system_reset_hook(self):
        """
        This is a function that gets called every time the dynamical system we are controlling gets episodically reset.
        Here, we use it to alert our dataset that the current trajectory has ended
        """
        if self.t < self.T0: self.dataset.reset()
        else: self.controller.system_reset_hook()
        pass
    
    def update(self, state: np.ndarray, cost: float, control: np.ndarray, next_state: np.ndarray, next_cost: float):
        if self.t < self.T0: 
            self.dataset.add_transition(state, cost, control, next_state, next_cost)
            return
        else: 
            return self.controller.update(state, cost, control, next_state, next_cost)
    
    def get_control(self, cost: float, obs: np.ndarray) -> np.ndarray:
        assert obs.shape == (self.obs_dim,), (obs.shape, self.obs_dim)
        
        self.t += 1
        if self.t < self.T0: return self.explorer.get_control(cost, obs)
        elif self.t == self.T0:
            logging.info('(OFFLINE SYSID) ending exploration at timestep {}'.format(self.t))
            
            # make the controller
            self.controller = self.make_controller(self.dataset)
            assert isinstance(self.controller, LiftedController), 'controller produced by `make_controller()` should be a LiftedController'
            assert self.controller.control_dim == self.control_dim, (self.controller.control_dim, self.control_dim)
            self.stats = self.controller.stats
            self.controller.t = self.t
            
        return self.controller.get_control(cost, obs)
    
    
# ------------------------------------------------------------------------------------
# --------------------------     ADAPTIVE REGRET   -----------------------------------
# ------------------------------------------------------------------------------------

def generate_death_times(n, min_lifetime, first_lifetime=None):
    def lifetime(i):
        l = 4
        while i % 2 == 0:
            l *= 2
            i /= 2
        return max(l + 1, min_lifetime)

    tod = np.arange(n)
    for i in range(1, n):
        tod[i] = i + lifetime(i)
    if first_lifetime is None: first_lifetime = min_lifetime
    tod[0] = first_lifetime  # lifetime not defined for 0
    return tod
    
    
class HardFTH(Controller):
    def __init__(self,
                 control_dim: int, obs_dim: int,
                 explorer: Explorer,  # for an initial exploration before spawining the first expert
                 make_controller: Callable[[Dataset], Controller],  # to make controllers
                 
                 eta: float,
                 T0: int,
                 spawn_every: int,
                 dataset_window: int,
                 min_lifetime: int,
                 ):
        super().__init__()
        assert control_dim == explorer.control_dim
        self.control_dim = control_dim
        self.state_dim = obs_dim
        self.make_controller = make_controller
        
        self.t = 0
        self.min_lifetime = min_lifetime
        self.tod = generate_death_times(10 * max(min_lifetime, T0), min_lifetime, first_lifetime=T0)
        self.spawn_every = spawn_every
        self.dataset_window = dataset_window
        
        self.dataset = Dataset(self.control_dim, self.state_dim)
        self.experts = {0: explorer}  # list of active experts
        self.eta = eta
        self.probs = {0: 1.}
        self.prev_selected_expert = 0  # idx of expert that we selected last
        
        self.stats = Stats()
        self.stats.register('states', obj_class=np.ndarray, shape=(self.state_dim,))
        self.stats.register('disturbances', obj_class=np.ndarray, shape=(self.state_dim,))
        self.stats.register('-K @ state', obj_class=np.ndarray, shape=(self.control_dim,))
        self.stats.register('M \cdot w', obj_class=np.ndarray, shape=(self.control_dim,))
        self.stats.register('M0', obj_class=np.ndarray, shape=(self.control_dim,))
        pass
        
    def system_reset_hook(self):
        self.dataset.reset()
        for e in self.experts.values(): e.system_reset_hook()
        pass
    
    def update(self, state: np.ndarray, cost: float, control: np.ndarray, next_state: np.ndarray, next_cost: float):
        self.dataset.add_transition(state, cost, control, next_state, next_cost)
        expert = self.experts[self.prev_selected_expert]
        expert.update(state, cost, control, next_state, next_cost)
        
        # collect the important stats from this step
        if expert.stats is not None:
            s = expert.stats._stats
            for k in {'disturbances',}.intersection(s.keys()):
                v = s[k].values
                if len(v) > 0: 
                    if not isinstance(v[-1], np.ndarray): v = np.array(v[-1])
                    else: v = v[-1]
                    self.stats.update(k, v, t=self.t)
        
        # decay probability of selected controller
        decay_val = max(np.exp(-self.eta * next_cost), 1e-3) if not (np.isnan(next_cost) or next_cost > 1e3) else 0.
        self.probs[self.prev_selected_expert] *= decay_val
        
        # kill whoever we have to
        kill_idx = np.where(self.tod == self.t)[0]
        if len(kill_idx) and kill_idx[0] in self.experts and len(self.experts) > 2:
            del self.experts[kill_idx[0]]
            del self.probs[kill_idx[0]]
        
        # renormalize, and spawn a new expert if necessary
        if self.t % self.spawn_every == 0 and self.t > 0:
            v = (1 - 1 / self.t) / sum(self.probs.values())
            for k in self.probs.keys(): self.probs[k] *= v
            self.probs[self.t] = 1 / self.t
            self.experts[self.t] = self.make_controller(self.dataset.get_tail(self.dataset_window))
        else:
            v = 1 / sum(self.probs.values())
            for k in self.probs.keys(): self.probs[k] *= v
        pass
    
    def get_control(self, cost: float, obs: np.ndarray) -> np.ndarray:
        assert obs.shape == (self.state_dim,), (obs.shape, self.state_dim)
        
        # handle time stuff
        self.t += 1
        for e in self.experts.values(): e.t = self.t
        if self.t >= len(self.tod): self.tod = generate_death_times(2 * self.t, self.min_lifetime)  # regen some more death times if we run out
        
        # pick a random expert and query it
        # self.prev_selected_expert = np.random.choice(list(self.probs.keys()), p=list(self.probs.values()))
        self.prev_selected_expert = list(self.probs.keys())[np.argmax(list(self.probs.values()))]
        expert = self.experts[self.prev_selected_expert]
        control = expert.get_control(cost, obs)
        
        # collect the most recent stats from the selected expert
        if expert.stats is not None:
            s = expert.stats._stats
            for k in {'states', '-K @ state', 'M0', 'M \cdot w'}.intersection(s.keys()):
                v = s[k].values
                if len(v) > 0: 
                    if not isinstance(v[-1], np.ndarray): v = np.array(v[-1])
                    else: v = v[-1]
                    self.stats.update(k, v, t=self.t)
        
        return control
    