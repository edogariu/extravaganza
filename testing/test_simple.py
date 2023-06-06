from controller import FloatHyperparameter
import numpy as np

a = 5

# def f(x):
#     return np.tanh(x - a) * np.tanh(x) + np.random.randn() * 0.01
# def grad_f(x):
#     return 2 * np.tanh(x - a) * (1 - np.tanh(x - a) * np.tanh(x - a))

def f(x):
    return (x - a) * (x - a) + np.random.randn() * 0.01
def grad_f(x):
    return 2 * (x - a)

alg_args = {
        'h': 20,
        'initial_value': 200,
        'initial_scale': 1,
        'cost_clip_size': 1.,
        # 'w_clip_size': 1,
        # 'M_clip_size': 1.,
        'method': 'FKM',
    }    
T = 7000

vals = []
x = FloatHyperparameter(**alg_args)
for _ in range(T):
    err = f(x)
    print(x)
    x.step(obj=err, grad_u=grad_f(x), B=0)
    # x.step(obj=err, B=0)
    vals.append(x.get_value())
    
print(x.M)
import matplotlib.pyplot as plt
from testing.utils import window_average
vals = window_average(vals, window_size=50)
plt.plot(range(len(vals)), vals)
plt.show()

