from typing import Callable, Dict
from collections import defaultdict
from copy import deepcopy
import tqdm

import jax.numpy as jnp

from extravaganza.controllers import Controller
from extravaganza.dynamical_systems import DynamicalSystem
from extravaganza.stats import Stats

def ylim(ax, left, right):
    _l, _r = ax.get_ylim()
    l = max(_l, left)
    r = min(_r, right)
    ax.set_ylim(l, r)
    return ax

def run_trial(controller: Controller, 
              system: DynamicalSystem, 
              T: int, 
              reset_condition: Callable[[int], bool] = lambda t: False,
              reset_seed: int = None,
              wordy: bool = True):
    
    # initial control
    control = controller.initial_control if hasattr(controller, 'initial_control') else jnp.zeros(controller.control_dim)  
    
    # run trial
    pbar = tqdm.trange(T) if wordy else range(T)
    for t in pbar:
        if reset_condition(t):
            print('reset!')
            system.reset(reset_seed)
            
        cost, state = system.interact(control)  # state will be `None` for unobservable systems
        control = controller.get_control(cost, state)
        
        if wordy: 
            postfix = {}
            if state is not None and state.shape == (1,): postfix['state'] = state.item()
            if control.shape == (1,): postfix['control'] = control.item()
            postfix['cost'] = cost
            pbar.set_postfix(postfix)
            
        
        if (state is not None and jnp.any(jnp.isnan(state))) or (cost > 1e20):
            print('WARNING: state {} or cost {} diverged'.format(state, cost))
            return None
    
    return Stats.concatenate((system.stats, controller.stats))


def run_experiment(make_system: Callable[[], DynamicalSystem],
                   make_controllers: Dict[str, Callable[[DynamicalSystem], Controller]],
                   num_trials: int,
                   T: int,
                   reset_condition: Callable[[int], bool] = lambda t: False,
                   reset_seed: int = None,
                   wordy: bool = True):
    
    stats = defaultdict(list)
    for t in range(num_trials):  # run many trials
        print('--------------------------------------------------')
        print('----------------- TRIAL {} -----------------------'.format(t))
        print('--------------------------------------------------\n')
        for k, controller_func in make_controllers.items():
            if wordy: print('testing {}'.format(k))
            system = make_system()
            controller = controller_func(system)
            s = run_trial(controller, system, T, reset_condition, reset_seed, wordy=wordy)
            if s is not None: stats[k].append(s)
            if wordy: print()
    if len(stats) == 0: 
        print('ERROR: none of the trials succeeded.')
        return None
    for k in stats.keys():  # aggregate stats over the trials
        stats[k] = Stats.aggregate(stats[k])
    
    return stats
