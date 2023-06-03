import warnings; warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import tqdm

# for multiprocessing -- NOTE you may need to `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` for this to work
from pathos.multiprocessing import _ProcessPool as Pool
from multiprocessing import Value
from os import cpu_count
import time

import numpy as np
import torch
import hypergrad

from hyperparameter import FloatHyperparameter
from optimizer import SGD
from testing.problems import PROBLEM_CLASSES, PROBLEM_ARGS
from testing.utils import window_average

def main():
    use_multiprocessing = False
    
    num_iters = 2000
    num_trials = 1
    step_every = 1
    eval_every = 2000
    reset_every = 400
    window_size = num_iters // 200
    seed = None
    
    problem = 'LR' # ['LR', 'MNIST MLP', 'MNIST CNN']

    lr_args = {
        'h': 5,
        'initial_value': 0.5,
        'interval': (0, 1),
        'rescale': True,
        'quadratic_term': 0,
        'w_clip_size': 0.1,
        # 'M_clip_size': 1e-9,
        # 'B_clip_size': 1,  
        # 'update_clip_size': 1,                                
        # 'cost_clip_size': 1,
        'method': 'FKM',
    }    
    momentum_args = {
        'h': 5,
        'initial_value': 0.2,
        'interval': (0, 0.99),
        'rescale': True,
        'quadratic_term': 0,
        # 'w_clip_size': 0.5,
        # 'M_clip_size': 1e-7,
        # 'B_clip_size': 0.5,
        # 'cost_clip_size': 0.5,
        'method': 'REINFORCE',
    }    
    optimizers = {
        'ours (lr)': lambda model: SGD(model.parameters(), lr=FloatHyperparameter(**lr_args), step_every=step_every),
        # 'ours (lr) +m': lambda model: SGD(model.parameters(), lr=FloatHyperparameter(**lr_args), momentum=0.9, step_every=step_every),
        # 'ours (m)': lambda model: SGD(model.parameters(), lr=0.05, momentum=FloatHyperparameter(**momentum_args), step_every=step_every),
        # 'ours (lr, m)': lambda model: SGD(model.parameters(), lr=FloatHyperparameter(**lr_args), momentum=FloatHyperparameter(**momentum_args), step_every=step_every),
        'SGD': lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        # 'SGD +m': lambda model: torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        # 'SGDHD': lambda model: hypergrad.SGDHD(model.parameters(), lr=0.2, hypergrad_lr=1e-7, weight_decay=1e-4),
        # 'SGDHD +m': lambda model: hypergrad.SGDHD(model.parameters(), lr=0.001, hypergrad_lr=0.001, momentum=0.9, weight_decay=1e-4),
        # 'ADAM': lambda model: torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5),
    }
    
    results = {}
    for opt_name, o in optimizers.items():
        probe_fns = {}
        if '(lr)' in opt_name: probe_fns['disturbances'] = lambda p: p.opt.param_groups[0]['lr'].ws[0]
        if '(m)' in opt_name: probe_fns['disturbances'] = lambda p: p.opt.param_groups[0]['momentum'].ws[0]
        results[opt_name] = do_trials(problem, o, seed, probe_fns, 
                                      num_trials, num_iters, eval_every, reset_every, use_multiprocessing)
        
    plot_results(results, window_size, problem)
    exit(0)
    
    
    
def plot_results(results, window_size, problem):
    fig, ax = plt.subplots(2, 2)
    STD_MULT = 1.0
    for opt_name, stats in results.items():
        # plot learning rates
        means = window_average(stats['lrs']['means'], window_size)
        stds = window_average(stats['lrs']['stds'], window_size)
        ax[0, 0].plot(stats['lrs']['ts'], means, label=opt_name)
        ax[0, 0].fill_between(stats['lrs']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
        
        # # plot momentum
        # means = window_average(stats['momenta']['means'], window_size)
        # stds = window_average(stats['momenta']['stds'], window_size)
        # ax[0, 1].plot(stats['momenta']['ts'], means, label=opt_name)
        # ax[0, 1].fill_between(stats['momenta']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
        
        # plot disturbances
        if 'ours' in opt_name:
            # means = window_average(stats['disturbances']['means'], window_size)
            # stds = window_average(stats['disturbances']['stds'], window_size)
            means = stats['disturbances']['means']
            stds = stats['disturbances']['stds']
            # ax[0, 1].plot(stats['disturbances']['ts'], means, label=opt_name)   
            # ax[0, 1].fill_between(stats['disturbances']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
            
            pos_t = []; pos = []; pos_std = []
            neg_t = []; neg = []; neg_std = []
            for t, m, s in zip(stats['disturbances']['ts'], means, stds):
                if m >= 0: pos_t.append(t); pos.append(m); pos_std.append(s)
                else: neg_t.append(t); neg.append(m); neg_std.append(s)
            pos_t, pos, pos_std, neg_t, neg, neg_std = np.array(pos_t), np.array(pos), np.array(pos_std), np.array(neg_t), np.array(neg), np.array(neg_std)

            ax[0, 1].plot(neg_t, neg, label=opt_name, color='red')
            ax[0, 1].fill_between(neg_t, neg - STD_MULT * neg_std, neg + STD_MULT * neg_std, alpha=0.5, color='red')
            ax[0, 1].plot(pos_t, pos, label=opt_name, color='blue')
            ax[0, 1].fill_between(pos_t, pos - STD_MULT * pos_std, pos + STD_MULT * pos_std, alpha=0.5, color='blue')
        
        # plot train losses
        means = window_average(stats['train_losses']['means'], window_size)
        stds = window_average(stats['train_losses']['stds'], window_size)
        # means = stats['train_losses']['means']
        # stds = stats['train_losses']['stds']
        ax[1, 0].plot(stats['train_losses']['ts'], means, label=opt_name)
        ax[1, 0].fill_between(stats['train_losses']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
        
        # plot val errors
        # means = window_average(stats['val_errors']['means'], window_size)
        # stds = window_average(stats['val_errors']['stds'], window_size)
        means = stats['val_errors']['means']
        stds = stats['val_errors']['stds']
        ax[1, 1].plot(stats['val_errors']['ts'], means, label=opt_name)
        ax[1, 1].fill_between(stats['val_errors']['ts'], means - STD_MULT * stds, means + STD_MULT * stds, alpha=0.5)
    
    ax[0, 0].set_title('{} learning rate'.format(problem))
    ax[0, 0].legend()
    ax[0, 0].set_ylim([0, 1])
    
    # ax[0, 1].set_title('{} momentum'.format(problem))
    # ax[0, 1].legend()
    # ax[0, 1].set_ylim([0, 1.])
    ax[0, 1].set_title('{} disturbances'.format(problem))
    ax[0, 1].legend()
    
    ax[1, 0].set_title('{} train losses'.format(problem))
    ax[1, 0].legend()
    ax[1, 0].set_ylim([0, 0.3 if 'MNIST' in problem else 0.03])
    
    ax[1, 1].set_title('{} val errors'.format(problem))
    ax[1, 1].legend()
    ax[1, 1].set_ylim([0, 0.3 if 'MNIST' in problem else 0.03])
    plt.show()
    pass


def init(args):
    ''' store the counter for later use '''
    global counter
    counter = args
    # sys.stdout = open(os.d evnull, 'w') 
    # sys.stderr = open(os.devnull, 'w')

def run_trial(problem, o, seed, probe_fns, num_iters, eval_every, reset_every, use_multiprocessing):
    problem = PROBLEM_CLASSES[problem](o, seed=seed, probe_fns=probe_fns, **PROBLEM_ARGS[problem])
    s = problem.train(num_iters, eval_every, reset_every, wordy=not use_multiprocessing)
    if use_multiprocessing:
        with counter.get_lock():
            counter.value += 1
    return s

def do_trials(problem, o, seed, probe_fns, num_trials, num_iters, eval_every, reset_every, use_multiprocessing):        
    if use_multiprocessing:        
        # split da pie
        counter = Value('i', 0)
        n_cpu = cpu_count()        

        with Pool(processes=n_cpu, initializer = init, initargs = (counter,)) as pool:
            stats = pool.starmap_async(run_trial, [(problem, o, i, probe_fns, num_iters, eval_every, reset_every, use_multiprocessing) for i in range(num_trials)])
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
        stats = [run_trial(problem, o, seed, probe_fns, num_iters, eval_every, reset_every, use_multiprocessing) for _ in range(num_trials)]
            
    ret = {}
    for k in stats[0].keys():
        r = {}
        m = np.stack([np.array(list(s[k].values())) for s in stats])
        r['ts'] = list(stats[0][k].keys())
        r['means'] = np.mean(m, axis=0)
        r['stds'] = np.std(m, axis=0)
        # r['means'] = np.median(m, axis=0)
        # r['stds'] = np.subtract(*np.percentile(m, [75, 25], axis=0))
        ret[k] = r
    
    return ret


if __name__ == '__main__':
   main()