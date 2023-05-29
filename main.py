from collections import defaultdict

from hyperparameter import FloatHyperparameter
from testing.problems import TorchLinearRegression, TorchMNIST
from testing.models import MLP
from testing.utils import window_average

if __name__ == '__main__':
    num_iters = 2000
    step_every = 2
    reset_every = 200
    seed = None
    
    alg_args = {
        'h': 5,
        'initial_value': 20,
        'initial_scale': 0.01,
        'nonnegative': True,
        'quadratic_term': 1,
        'w_clip_size': 1.,
        # 'M_clip_size': 1.,
        'B_clip_size': 1,
        'cost_clip_size': 1,
        'method': 'FKM',
    }    
    baseline_args = {
        'SGD': {'lr': 0.1}
    }
    
    problem_class = TorchLinearRegression 
    problem_args = {
        'batch_size': 64,
        'n_features': 100,
        'n_informative': 30,
        'n_samples': 1000,
        'noise': 0.1,  # std dev of noise
    }
    
    # problem_class = TorchMNIST
    # problem_args = {
    #     'batch_size': 64,
    #     'make_model': lambda: MLP(layer_dims=[int(28 * 28), 100, 10]),
    # }

    # make problem with eta_0
    lr = FloatHyperparameter(**alg_args)
    opt_args = {'ours': {'lr': lr}}
    opt_args.update(baseline_args)
    problem = problem_class(opt_args=opt_args, seed=seed, **problem_args)
    
    vals = [lr.get_value()]
    disturbances = []
    errors = defaultdict(list)
    
    # repeat
    for n_step in range(num_iters // step_every):
        if (n_step * step_every) % reset_every == 0:
            problem.reset()
            
        try:
            # train until next step
            ret = problem.train(step_every)
            
            # collect errors
            for k, v in ret['errors'].items():
                errors[k].append(v)
                
            # step
            print('f(x_{}) = {}'.format(n_step, errors['ours'][-1]))
            lr.step(obj=errors['ours'][-1], grad_u=ret['grad_lr'], B=-ret['grad_mag'])
            # lr.step(obj=errors['ours'][-1], B=-ret['grad_mag'])

            vals.append(lr.get_value())
            if len(lr.ws) > 0: disturbances.append(lr.ws[0])
        except KeyboardInterrupt:
            print('Catching keyboard interrupt!')
            break
        
    import matplotlib.pyplot as plt
    window_size = 30

    fig, ax = plt.subplots(1, 3)
    vals = window_average(vals, window_size)
    ax[0].plot(range(len(vals)), vals, label='ours')
    ax[0].set_title('learning rate')
    
    for k, v in errors.items(): ax[1].plot(range(len(v)), v, label=k)
    ax[1].set_title('errors')
    ax[1].set_ylim([0, 0.3])
    ax[1].legend()
    
    disturbances = window_average(disturbances, window_size)
    ax[2].plot(range(len(disturbances)), disturbances)
    ax[2].set_title('disturbances')
    plt.show()
    exit(0)
