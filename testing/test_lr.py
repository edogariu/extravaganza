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

def main():
    from optimizer import SGD
    from dynamical_systems import COCO, LinearRegression, MNIST
    
    use_multiprocessing = False
    num_iters = 10000
    reset_every = 50000
    num_trials = 1
    controller_args = {
        'h': 5,
        'initial_value': 0.5,
        'bounds': (0, 1),
        # 'w_clip_size': 1,
#         'M_clip_size': 1e-9,
        # 'B_clip_size': 1,  
        # 'update_clip_size': 1,                                
        # 'cost_clip_size': 1,
        'method': 'REINFORCE',
    }       
    
    make_opt = lambda model: SGD(model.parameters(), lr=FloatController(**controller_args), step_every=1)
    probe_fns = {'disturbances': lambda system, controller: system.opt.param_groups[0]['lr'].ws[0],
                 'ds': lambda system, controller: system.opt.param_groups[0]['lr'].update.d,
                 }
    
    system = LinearRegression(make_opt, dataset='generated', probe_fns=probe_fns)
    system_name = 'Linear Regression'
    
    # system = MNIST(make_opt, probe_fns=probe_fns)
    # system_name = 'MNIST'
    
    stats = run(system, num_iters, num_trials, controller_args=None, use_multiprocessing=use_multiprocessing, reset_every=reset_every)
    results = {'GPC': stats}
    plot_results(results, num_iters // 200, system_name)
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
            system.reset_episode()
        if step_controller:
            f = system.interact(controller)
            if t % step_every == 0: controller.step(obj=f)
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
            stats = pool.starmap(run_interaction_loop, [(system, num_iters, reset_every, controller_args, step_every, use_multiprocessing) for _ in range(num_trials)])
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
        # plot lrs
        means = window_average(stats['lrs']['means'], window_size)
        stds = window_average(stats['lrs']['stds'], window_size)
        ax[0, 0].plot(stats['lrs']['ts'], means, label=controller_name)
        ax[0, 0].fill_between(stats['lrs']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
        
        # plot disturbances
        if 'disturbances' in stats:
            # means = window_average(stats['disturbances']['means'], window_size)
            # stds = window_average(stats['disturbances']['stds'], window_size)
            means = stats['disturbances']['means']
            stds = stats['disturbances']['stds']
            ax[0, 1].plot(stats['disturbances']['ts'], means, label=controller_name)   
            ax[0, 1].fill_between(stats['disturbances']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
            
        # plot objective vs time
        # means = window_average(stats['train_losses']['means'], window_size)
        # stds = window_average(stats['train_losses']['stds'], window_size)
        means = stats['train_losses']['means']
        stds = stats['train_losses']['stds']
        ax[1, 0].plot(stats['train_losses']['ts'], means, label=controller_name)
        ax[1, 0].fill_between(stats['train_losses']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
        
        if 'ds' in stats:
            means = stats['ds']['means']
            stds = stats['ds']['stds']
            ax[1, 1].plot(stats['ds']['ts'], means, label=controller_name)
            ax[1, 1].fill_between(stats['ds']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
        
    ax[0, 0].set_title('{} learning rates'.format(system_name))
    ax[0, 0].legend()
    
    ax[0, 1].set_title('{} disturbances'.format(system_name))
    ax[0, 1].legend()
    
    ax[1, 0].set_title('{} train losses'.format(system_name))
    ax[1, 0].legend()
    
    ax[1, 1].set_title('{} d'.format(system_name))
    plt.show()
    pass

if __name__ == '__main__':
    main()
    