from copy import deepcopy
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# for multiprocessing -- NOTE you may need to `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` for this to work
from pathos.multiprocessing import _ProcessPool as Pool
from multiprocessing import Value
from os import cpu_count
import time

from dynamical_systems import DynamicalSystem
from controller import FloatController
from testing.utils import window_average

a = 4
c = 1

# def f(x):
#     return c * np.tanh(x - a) * np.tanh(x) + np.random.randn() * 0.01 * c
# def grad_f(x):
#     return c * 2 * np.tanh(x - a) * (1 - np.tanh(x - a) * np.tanh(x - a))

def f(x):
    return c * (x - a) * (x - a) + np.random.randn() * 0.01 * c
def grad_f(x):
    return c * 2 * (x - a)


def main():
    from dynamical_systems import TestSystem
    
    use_multiprocessing = False
    num_iters = 5000
    num_trials = 4
    controller_args = {
        'h': 5,
        'initial_value': 0.5,
        'bounds': (-5, 5),
        # 'w_clip_size': 1,
    #         'M_clip_size': 1e-9,
        # 'B_clip_size': 1,  
        # 'update_clip_size': 1,                                
        # 'cost_clip_size': 1,
        'method': 'FKM',
    }    
    
    probe_fns = {'disturbances': lambda system, controller: controller.ws[0]}
    system = TestSystem(problem_fn=f, grad_fn=grad_f, probe_fns=probe_fns)
    # system = TestSystem(problem_fn=f, grad_fn=None, probe_fns=probe_fns)
    stats = run(system, num_iters, num_trials, controller_args=controller_args, use_multiprocessing=use_multiprocessing)
    results = {'GPC': stats}
    plot_results(results, num_iters // 200, 'test')
    exit(0)
    
def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args
    # sys.stdout = open(os.devnull, 'w') 
    # sys.stderr = open(os.devnull, 'w')

def run_interaction_loop(system: DynamicalSystem, 
                         num_iters: int,
                         reset_every: int=None,
                         controller_args=None,  # if `controller_args is None`, we assume that interacting with the system steps the controller
                         step_every: int=1,
                         use_multiprocessing: bool=False,
                         ):
    
    system.reset()
    
    step_controller = controller_args is not None
    if step_controller:  # if we need to explicitly make the controller
        if hasattr(system, 'get_init'):  # in case the system has its own specified init and bounds
            initial_value, bounds = system.get_init()
            controller_args['initial_value'] = initial_value
            controller_args['bounds'] = bounds
        controller = FloatController(**controller_args)
    
    pbar = range(num_iters) if use_multiprocessing else tqdm.trange(num_iters)
    for t in pbar:
        if reset_every is not None and t % reset_every == 0: 
            system.reset()
        if step_controller:
            f = system.interact(controller)
            # if t % step_every == 0: controller.step(obj=f, B=0)
        else:
            system.interact(None)
    if use_multiprocessing:
        with counter.get_lock():
            counter.value += 1    
    d = deepcopy(system.stats)
    return d

def run(system: DynamicalSystem, 
        num_iters: int,
        num_trials: int,
        reset_every: int=None,
        controller_args=None,  # if `controller_args is None`, we assume that interacting with the system steps the controller
        step_every: int=1,
        use_multiprocessing: bool=False,):
    
    if use_multiprocessing:        
        # split da pie
        counter = Value('i', 0)
        n_cpu = cpu_count()        

        with Pool(processes=n_cpu, initializer = init, initargs = (counter,)) as pool:
            stats = pool.starmap_async(run_interaction_loop, [(system, num_iters, reset_every, controller_args, step_every, use_multiprocessing) for _ in range(num_trials)])
            prev_count = counter.value - 1
            with tqdm.tqdm(total=num_trials) as pbar:
                while counter.value < num_trials:
                    if counter.value != prev_count:
                        pbar.update(counter.value - prev_count)
                        prev_count = counter.value
                    else:
                        time.sleep(0.0001)
            pool.close()
            pool.join()
            stats = stats.get()
    else:
        stats = [run_interaction_loop(system, num_iters, reset_every, controller_args, step_every, use_multiprocessing) for _ in range(num_trials)]
            
    ret = {}
    for k in stats[0].keys():
        r = {}
        m = np.stack([np.array(list(s[k].values())) for s in stats])
        r['ts'] = list(stats[0][k].keys())
        r['means'] = np.mean(m, axis=0)
        r['stds'] = np.std(m, axis=0)
        r['m'] = m
        ret[k] = r
    
    return ret

def plot_results(results, window_size: int, system_name: str):
    fig, ax = plt.subplots(2, 2)
    STD_MULT = 1.0
    for controller_name, stats in results.items():
        # plot controls
        means = window_average(stats['controls']['means'], window_size)
        stds = window_average(stats['controls']['stds'], window_size)
        ax[0, 0].plot(stats['controls']['ts'], means, label=controller_name)
        ax[0, 0].fill_between(stats['controls']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
        if 'optimal_control' in stats:
            ax[0, 0].plot(stats['controls']['ts'], [a for _ in stats['controls']['ts']], label='optimal')
        
        # plot disturbances
        if 'disturbances' in stats:
            # means = window_average(stats['disturbances']['means'], window_size)
            # stds = window_average(stats['disturbances']['stds'], window_size)
            means = stats['disturbances']['means']
            stds = stats['disturbances']['stds']
            ax[0, 1].plot(stats['disturbances']['ts'], means, label=controller_name)   
            ax[0, 1].fill_between(stats['disturbances']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
            
        # plot objective vs time
        # means = window_average(stats['objectives']['means'], window_size)
        # stds = window_average(stats['objectives']['stds'], window_size)
        means = stats['objectives']['means']
        stds = stats['objectives']['stds']
        ax[1, 0].plot(stats['objectives']['ts'], means, label=controller_name)
        ax[1, 0].fill_between(stats['objectives']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
        
        # plot objective vs control
        us, fs = stats['gt_controls']['m'].squeeze()[0], stats['gt_values']['m'].squeeze()[0]
        ax[1, 1].plot(us, fs)
        
    ax[0, 0].set_title('{} controls'.format(system_name))
    ax[0, 0].legend()
    
    ax[0, 1].set_title('{} disturbances'.format(system_name))
    ax[0, 1].legend()
    
    ax[1, 0].set_title('{} objectives'.format(system_name))
    ax[1, 0].legend()
    
    ax[1, 1].set_title('{} objectives vs controls'.format(system_name))
    plt.show()
    pass

if __name__ == '__main__':
    main()
    
