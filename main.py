from hyperparameter import FloatHyperparameter
from optimization_problems import TorchLinearRegression, TorchMNISTMLP

if __name__ == '__main__':
    # heres how it should go. this is all that is needed
    T = 40
    initial_lr = 0.01
    seed = None  # 0

    # for LINEAR REGRESSION
    num_iters = 400
    batch_size = 64
    problem_class = TorchLinearRegression
    args = {
        'n_features': 30,
        'n_informative': 10,
        'n_samples': 1000,
        'noise': 0.1,  # std dev of noise
    }

    # # for MNIST with MLP
    # num_iters = 200
    # batch_size = 64
    # problem_class = TorchMNISTMLP
    # args = {  # for MNIST with MLP
    #     'layer_dims': [int(28 * 28), 100, 100, 100, 100, 10], 
    #     # 'layer_dims': [int(28 * 28), 100, 10],               
    # }

    # make problem with eta_0
    lr = FloatHyperparameter(h=5, initial_value=initial_lr)
    problem = problem_class(lr, **args)
    
    # repeat
    for t in range(T + 1):
        test_err = problem.train(num_iters, batch_size)
        print('o_{} = {}'.format(t, test_err))
        lr.step(o=test_err)
        problem.reset()
