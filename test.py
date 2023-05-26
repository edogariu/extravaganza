from new_hyperparameter import FloatHyperparameter
import numpy as np

a = 3

# def f(x):
#     return np.tanh(x - a) * np.tanh(x) + np.random.randn() * 0.01
# def grad_f(x):
#     return 2 * np.tanh(x - a) * (1 - np.tanh(x - a) * np.tanh(x - a))

def f(x):
    return (x - a) * (x - a) + np.random.randn() * 0.01
def grad_f(x):
    return 2 * (x - a)

alg_args = {
        'h': 5,
        'initial_value': -1,
        'initial_scale': 1,
        'w_clip_size': 0.1,
        # 'M_clip_size': 1.,
        'method': 'REINFORCE',
    }    
T = 10000

x = FloatHyperparameter(**alg_args)
for _ in range(T):
    err = f(x)
    print(err)
    x.step(obj=err, grad_u=grad_f(x), B=0)
    # x.step(obj=err, B=0)


