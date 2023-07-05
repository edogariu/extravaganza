import logging
from typing import Any, Callable, Dict
from collections import defaultdict
import tqdm
import random
import time
import matplotlib.pyplot as plt
import dill
from dill.source import getsource

# for multiprocessing
import os
from multiprocessing import Manager, Value
# from multiprocessing import set_start_method; set_start_method('spawn'); logging.info('(EXPERIMENT): set multiprocessing start method to spawn')
from pathos.multiprocessing import ProcessPool as Pool

import numpy as np
import jax.numpy as jnp

from extravaganza.controllers import Controller
from extravaganza.dynamical_systems import DynamicalSystem
from extravaganza.stats import Stats
from extravaganza.utils import set_seed, get_color

def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args
    # sys.stdout = open(os.devnull, 'w') 
    # sys.stderr = open(os.devnull, 'w')

class Experiment:
    def __init__(self, name: str):
        self.name = name
        self.get_args = None
        self.stats = None
        pass
    
    def __call__(self, *args, **kwargs):
        if self.stats is not None:
            logging.error('(EXPERIMENT): experiment named {} has already been run!'.format(self.name))
        
        if len(args) == 1:
            get_args = args[0]
            assert isinstance(get_args, Callable)
            self.get_args = get_args
            
        if self.get_args is not None:
            experiment_args = self.get_args()
            stats = self._run_experiment(**experiment_args)
        else:
            logging.warn('(EXPERIMENT): you are providing the experiment args directly and not through a getter function, which means we wont be able to print the args later :(')
            stats = self._run_experiment(*args, **kwargs)
        self.stats = stats
        return stats
    
    def print_args(self):
        if self.get_args is None:
            logging.error('(EXPERIMENT): cannot print the experiment args because the object was not called with a getter function')
            raise Exception()
        print('----------------------------------------------------------------------------------------------------')
        print('----------------------------  EXPERIMENT ARGUMENTS FOR {}  -----------------------------------------'.format(self.name))
        print('----------------------------------------------------------------------------------------------------')
        print()
        print(getsource(self.get_args))
        pass
    
    def save(self, filename: str):
        if self.stats is None: logging.warn('(EXPERIMENT): trying to save before running the experiment?')
        if self.get_args is None: logging.warn('(EXPERIMENT): saving experiment without providing a getter function. some hyperparams may be uncheckable after loading. Can be fixed with `self.get_args = ...`')
        with open(filename, 'wb') as f:
            dill.dump(self, f)
        logging.info('(EXPERIMENT) saved experiment to `{}`'.format(filename))
        return self
    
    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            exp = dill.load(f)
        logging.info('(EXPERIMENT) loaded experiment from `{}`'.format(filename))
        return exp
    
    
    def _run_trial(self,
                  make_system: Callable[[], DynamicalSystem],
              make_controller: Callable[[DynamicalSystem], Controller],
              T: int, 
              reset_condition: Callable[[int], bool] = lambda t: False,
              reset_seed: int = None,
              render_every: int = None,
              append_list: list = None,
              id: str = None,  # for labeling in the renderer
              seed: int = None) -> Stats:
        try:
            set_seed(meta_seed=seed)
            
            # make system and controller
            system = make_system()
            controller = make_controller(system)
            
            # initial control
            control = controller.initial_control if hasattr(controller, 'initial_control') else jnp.zeros(controller.control_dim)  
            
            # for rendering
            if render_every is None:
                def render(t, cost, state): return  # no-op
            else:
                costs = []
                states = []
                ax = plt.gca()
                # _, ax = plt.subplots(figsize=(10, 6))
                color = get_color(id)
                line, = ax.plot([], [], '', color=color, lw=1.5, label=id)
                point, = ax.plot([], [], 'o', color=color)
                ax.set_xlabel('position')
                ax.set_ylabel('cost')
                ax.legend()
                
                def render(t, cost, state):
                    costs.append(cost)
                    if state.shape == (1,): states.append(state.item())
                    else:
                        logging.error('(EXPERIMENT): cannot render when state is {}-dimensional'.format(state.shape))
                        raise RuntimeError()
                    
                    if t % render_every != 0: return
                    
                    line.set_data(states, costs)
                    point.set_data([states[-1],], [costs[-1],])
                    maxs = np.max(states)
                    mins = np.min(states)
                    maxc = np.max(costs)
                    minc = np.min(costs)
                    ds = max(maxs - mins, 1e-6)
                    dc = max(maxc - minc, 1e-6)
                    margin = 0.05
                    ax.set_xlim(mins - margin * ds, maxs + margin * ds)
                    ax.set_ylim(minc - margin * dc, maxc + margin * dc)
                    plt.pause(0.0001)
                    return
                    
            # run trial
            pbar = tqdm.trange(T) if append_list is None else range(T)
            for t in pbar:
                if reset_condition(t):
                    logging.info('(EXPERIMENT): reset!')
                    system.reset(reset_seed)
                    
                cost, state = system.interact(control)  # state will be `None` for unobservable systems
                control = controller.get_control(cost, state)
                    
                render(t, cost, state)
                
                if append_list is None: 
                    postfix = {}
                    if state is not None and state.shape == (1,): postfix['state'] = state.item()
                    if control.shape == (1,): postfix['control'] = control.item()
                    postfix['cost'] = cost
                    pbar.set_postfix(postfix)
                
                if (state is not None and jnp.any(jnp.isnan(state))) or (cost > 1e20):
                    logging.error('(EXPERIMENT): state {} or cost {} diverged'.format(state, cost))
                    if append_list is not None: 
                        with counter.get_lock():
                            counter.value += 1
                    return None
                
            stats = Stats.concatenate((system.stats, controller.stats))
            if append_list is not None: 
                lock = append_list._mutex
                lock.acquire()
                append_list.append(stats)
                lock.release()
                with counter.get_lock():
                    counter.value += 1
        except Exception as e:
            logging.error('(EXPERIMENT): {}'.format(e))
            if append_list is not None: 
                with counter.get_lock():
                        counter.value += 1
            return None
        
        return stats

    def _run_experiment(self,
                       make_system: Callable[[], DynamicalSystem],
                   make_controllers: Dict[str, Callable[[DynamicalSystem], Controller]],
                   num_trials: int,
                   T: int,
                   reset_condition: Callable[[int], bool] = lambda t: False,
                   reset_seed: int = None,
                   use_multiprocessing: bool = False,
                   render_every: int = None):
    
        assert not (use_multiprocessing and render_every is not None), 'cannot render while multiprocessing'
        
        start_time = time.perf_counter()
        
        if use_multiprocessing:
            global counter
            
            man = Manager()
            stats_dict = {}
            args = defaultdict(list)
            counter = Value('i', 0)
            n = num_trials * len(make_controllers)
            for t in range(num_trials):
                for k, controller_func in make_controllers.items():
                    if k not in stats_dict: stats_dict[k] = man.list()
                    for i, v in enumerate([make_system, controller_func, T, reset_condition, reset_seed, render_every, stats_dict[k], k, random.randint(0, 10000)]): 
                        args[i].append(v)
                
            ncpu = os.cpu_count()
            logging.info('(EXPERIMENT): multiprocessing with {} cpus!'.format(ncpu))
            with Pool(processes = ncpu, initializer = init, initargs = (counter, )) as pool:
                prev = 0
                pool.amap(self._run_trial, *args.values())
                with tqdm.tqdm(total=n) as pbar:
                    while prev < n:
                        curr = counter.value
                        if curr != prev:
                            pbar.update(curr - prev)
                            prev = curr
                        else:
                            time.sleep(0.001)
            stats = {k: Stats.aggregate(v) for k, v in stats_dict.items() if len(v) > 0}
        else:    
            stats = defaultdict(list)
            for t in range(num_trials):  # run many trials
                logging.info('(EXPERIMENT) --------------------------------------------------')
                logging.info('(EXPERIMENT) ----------------- TRIAL {} -----------------------'.format(t))
                logging.info('(EXPERIMENT) --------------------------------------------------\n')
                for k, controller_func in make_controllers.items():
                    logging.info('(EXPERIMENT): testing {}'.format(k))
                    s = self._run_trial(make_system, controller_func, T, reset_condition, reset_seed, render_every=render_every, id=k)
                    if s is not None: stats[k].append(s)
                    logging.info('')
            if len(stats) == 0:
                logging.error('(EXPERIMENT): none of the trials succeeded.')
                return None
            for k in stats.keys():  # aggregate stats over the trials
                stats[k] = Stats.aggregate(stats[k])

        logging.info('(EXPERIMENT) done! The entire experiment took {} seconds'.format(time.perf_counter() - start_time))
        return stats
