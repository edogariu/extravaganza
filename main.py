from new_hyperparameter import FloatHyperparameter
from optimization_problems import TorchLinearRegression, TorchMNISTMLP

if __name__ == '__main__':
    # heres how it should go. this is all that is needed
    T = 3000
    reset_every = 500
    alg_args = {
        'h': 20,
        'initial_value': 0.3,
        'initial_scale': 0.1,
        'w_clip_size': 0.2,
        # 'M_clip_size': 1.,
        'method': 'FKM',
    }    
    
    baseline_args = {
        'SGD': {'lr': 0.2}
    }

    # # for LINEAR REGRESSION
    # num_iters = 4
    # batch_size = 64
    # problem_class = TorchLinearRegression
    # args = {
    #     'n_features': 30,
    #     'n_informative': 10,
    #     'n_samples': 1000,
    #     'noise': 0.1,  # std dev of noise
    # }

    # for MNIST with MLP
    num_iters = 4
    batch_size = 64
    problem_class = TorchMNISTMLP
    args = {  # for MNIST with MLP
        # 'layer_dims': [int(28 * 28), 100, 100, 100, 100, 10], 
        'layer_dims': [int(28 * 28), 100, 10],               
    }

    # make problem with eta_0
    lr = FloatHyperparameter(**alg_args)
    problem = problem_class(lr, baseline_args=baseline_args, **args)
    
    vals = [lr.get_value()]
    errors = []
    baseline_errors = {k: [] for k in baseline_args.keys()}
    
    # repeat
    for t in range(T + 1):
        if t % reset_every == 0:
            problem.reset()
        try:
            err, baseline_errs, grad_lr, grad_mag = problem.train(num_iters, batch_size)
            print('f(x_{}) = {}'.format(t, err))
            # lr.step(obj=err, grad_u=grad_lr, B=-grad_mag)
            lr.step(obj=err, B=-grad_mag)
            # problem.reset()

            vals.append(lr.get_value())
            errors.append(err)
            for k, l in baseline_errors.items(): l.append(baseline_errs[k])
            
        except KeyboardInterrupt:
            print('Catching keyboard interrupt!')
            break
        
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(1, 2)
    window_size = 30
    vals = np.convolve(vals, np.ones(window_size) / window_size)
    ax[0].plot(range(len(vals)), vals, label='our method')
    ax[0].set_title('learning rate')
    
    ax[1].plot(range(len(errors)), errors, label='our method')
    for k, v in baseline_errors.items(): ax[1].plot(range(len(v)), v, label=k)
    ax[1].set_title('errors')
    ax[1].legend()
    plt.show()
    exit(0)
