from hyperparameter import FloatHyperparameter

def f(x):
    return x * x

initial_x = 0.2
T = 100
h = 5

x = FloatHyperparameter(h=h, initial_value=initial_x)
for _ in range(T):
    err = f(x)
    print(err)
    x.step(o=err)


