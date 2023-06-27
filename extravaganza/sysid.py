import numpy as np
import jax.numpy as jnp

from extravaganza.utils import sample, set_seed, jkey, least_squares, opnorm, dare_gain

class SysID:
    """
    Determine `A` and `B` matrices for a LDS `x_{t+1} = A @ x_t + B @ u_t
    """
    def __init__(self,
                 method: str,  # must be one of ['moments', 'regression']
                 control_dim: int,
                 state_dim: int,
                 scale: float,
                 seed: int = None):
        set_seed(seed)  # for reproducibility
        
        assert method in ['moments', 'regression']
        self.method = method
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.scale = scale
        
        self.control_history = []
        self.state_history = []
        self.eps_history = []
        
        self.t = 1
        self.A = self.B = None
        pass
    
    
    def perturb_control(self,
                        state: jnp.ndarray,
                        control: jnp.ndarray=None):
        """
        if `control` is not None, we perturb around this control only when 'HAZAN' mode. otherwise, we don't perturb
        """
        assert state.shape == (self.state_dim,)
        
        if control is None or self.method == 'moments':
            control = control if control is not None else jnp.zeros(self.control_dim)
            eps = sample(jkey(), (self.control_dim, ))  # random direction
            control = control + self.scale * eps
        else:
            eps = jnp.zeros(self.control_dim)

        self.control_history.append(control)
        self.state_history.append(state)
        self.eps_history.append(eps)
        self.t += 1
        return control
    
    
    def sysid(self):
        assert self.t > 1
        
        if self.A is not None: return self.A, self.B
        
        if self.method == 'moments':
            k = int(0.15 * self.t)

            states = jnp.array(self.state_history)
            eps = jnp.array(self.eps_history)

            # prepare vectors and retrieve B
            scan_len = self.t - k - 1 # need extra -1 because we iterate over j = 0, ..., k
            N_j = jnp.array([jnp.dot(states[j: j + scan_len].T, eps[:scan_len]) for j in range(k + 1)]) / scan_len
            B = N_j[0] # jnp.dot(states[1:].T, eps[:-1]) / (self.t - 1)

            # retrieve A
            C_0, C_1 = N_j[:-1], N_j[1:]
            C_inv = jnp.linalg.inv(jnp.tensordot(C_0, C_0, axes=((0, 2), (0, 2))) + 1e-3 * np.identity(self.state_dim))
            A = jnp.tensordot(C_1, C_0, axes=((0, 2), (0, 2))) @ C_inv

        elif self.method == 'regression':
            # transform x and u into regular numpy arrays for least squares
            states = np.array(self.state_history)
            controls = np.array(self.control_history)

            # regression on A and B jointly
            A, B = least_squares(states, controls, max_opnorm=1.)
                
        self.A, self.B = A, B
        return A, B
    
    
    def dynamics(self,
                 state: jnp.ndarray,
                 control: jnp.ndarray):
        assert state.shape == (self.state_dim,) and control.shape == (self.control_dim,)
        A, B = self.sysid()  # make sure we have an estimate first
        
        return A @ state + B @ control
        
    
    def get_lqr(self):
        A, B = self.sysid()  # make sure we have an estimate first
            
        # compute stabilizing controller for squared costs
        Q = jnp.eye(self.state_dim); print('solving DARE with unconstrained Q')
        # Q = jnp.zeros((self.state_dim, self.state_dim)).at[-1, -1].set(1.); print('solving DARE with constrained Q')
        R = jnp.eye(self.control_dim)  # heuristic to weight state vs control
        
        try:  # TODO check here whether `K` stabilizes our system or not
            K = dare_gain(A, B, Q, R)  # solve the ricatti equation to compute LQR gain
            print('||A||_op = {}     ||B||_F {}         ||A-BK||_op = {}'.format(opnorm(A), jnp.linalg.norm(B, 'fro'), opnorm(A - B @ K)))
        except Exception as e:
            print('WARNING: K diverged', e)
            K = jnp.zeros((self.control_dim, self.state_dim))

        return K
    