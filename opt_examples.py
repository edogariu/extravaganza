import jax
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, 
                initial_lr: float, 
                ):
        """
        forms a linear regression problem.
        This can be used to get observations (and mayebe even predictions) and also to see if updating the learning rates is helfpful

        Parameters
        ----------
        initial_lr : float
            learning rate to start with
        """
        self.lr = initial_lr
    
    def train(self,
              num_epochs: int=100, 
              num_features: int=100, 
              num_informative_features: int=10,
              num_samples: int=300, 
              noise_std: float=0.1,) -> float:
        """
        returns final validation loss
        
        Parameters
        ----------
        num_epochs : int, optional
            number of epochs to iterate over for each observation, by default 100
        num_features : int, optional
            number of features for the regression problem, by default 100
        num_informative_features : int, optional
            number of features that are used to predict output, by default 10
        num_samples : int, optional
            number of samples for regression, by default 300
        noise_std : float, optional
            std dev of centered Gaussian noise applied to output
        """

        # DATASET
        X, y = make_regression(n_features=num_features, n_samples=num_samples, n_informative=num_informative_features, noise=noise_std)
        X, X_test, y, y_test = train_test_split(X, y)

        # MODEL
        params = {
            'w': jnp.zeros(X.shape[1:]),
            'b': 0.
        }

        def loss_fn(params, X, y):  # MSE
            err = jnp.dot(X, params['w']) + params['b'] - y
            return jnp.mean(jnp.square(err))

        grad_fn = jax.grad(loss_fn)
        losses = []
        for _ in range(num_epochs):
            # compute loss
            loss = loss_fn(params, X_test, y_test)
            losses.append(loss)

            # GD update
            grads = grad_fn(params, X, y)
            params = jax.tree_map(lambda p, g: p - self.lr * g, params, grads)
        return losses[-1]
    
    def update_lr(self, lr):
        print('Updated learning rate from {} to {}!'.format(self.lr, lr))
        self.lr = lr


if __name__ == '__main__':
    # heres how it should go
    
    from hyperparameter import FloatHyperparameter

    T = 0
    initial_lr = 0.2
    train_args = {
        'num_epochs': 400,
        'num_features': 100,
        'num_informative_features': 10,
        'num_samples': 300,
        'noise_std': 0.1,
    }
    
    # make problem with eta_0
    lr = FloatHyperparameter(h=5, initial_value=initial_lr)
    regression = LinearRegression(initial_lr=lr)
    
    # repeat
    for t in range(T + 1):
        val_err = regression.train(**train_args)
        print('o_{} = {}'.format(t, val_err))
        # lr.step(o=val_err)
    