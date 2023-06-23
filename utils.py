import random
import scipy.optimize as optimize
from scipy.linalg import solve_discrete_are
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

SAMPLING_METHOD = 'sphere'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def opnorm(X):
    if isinstance(X, torch.Tensor):
        return torch.amax(torch.abs(torch.linalg.eigvals(X)))
    elif isinstance(X, np.ndarray):
        return np.amax(np.abs(np.linalg.eigvals(X)))
    elif isinstance(X, jnp.ndarray):
        return jnp.amax(jnp.abs(jnp.linalg.eigvals(X))).item()
    else:
        raise NotImplementedError(X.__class__)
    
def dare_gain(A, B, Q, R):
    if isinstance(A, torch.Tensor):
        P = Riccati.apply(A, B, Q, R)
        K = torch.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # compute LQR gain
    elif isinstance(A, np.ndarray):
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # compute LQR gain
    elif isinstance(A, jnp.ndarray):
        P = solve_discrete_are(A, B, Q, R)
        K = jnp.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # compute LQR gain
    return K

def least_squares(xs: np.ndarray, 
                  us: np.ndarray, 
                  max_opnorm: float = None):
    
    """
    runs least squares to find A, B s.t.
        `A @ x_{t} + B @ u_{t} = x_{t+1}`
    """
    
    ds = xs.shape[-1]  # state dim
    du = us.shape[-1]  # control dim
    
    x_in = xs[:-1]
    x_out = xs[1:]
    u_in = us[:-1]
    
    if max_opnorm is None:
        A_B = np.linalg.lstsq(np.hstack((x_in, u_in)), x_out)[0]
        A, B = A_B[:ds].T, A_B[ds:].T
    else:
        print('constraining operator norm of `A` to be <= {}'.format(max_opnorm))
        A_B_shape = (ds, ds + du)
        def A_opnorm(t):
            A = t.reshape(*A_B_shape)[:, :ds]  # only grab A
            lam = np.amax(np.abs(np.linalg.eigvals(A)))
            return lam
        
        x0 = np.zeros(A_B_shape)
        x_in, x_out, u_in = np.expand_dims(x_in, axis=-1), np.expand_dims(x_out, axis=-1), np.expand_dims(u_in, axis=-1)
        nlc = optimize.NonlinearConstraint(A_opnorm, 0, max_opnorm)
        def obj(t):
            t = t.reshape(*A_B_shape)
            _A, _B = t[:, :ds], t[:, ds:]
            return ((_A @ x_in + _B @ u_in - x_out) ** 2).sum(axis=1).mean()
    
        omin = optimize.minimize(obj, x0.reshape(-1), method='SLSQP', options={'ftol': 1e-10,'disp': False}, constraints=nlc)
        A_B = omin['x'].reshape(A_B_shape)
        A, B = A_B[:, :ds], A_B[:, ds:]
        
    return jnp.array(A), jnp.array(B)


# --------------------------------------------------------------------------------
# teeny bit of hacking to make the jax PRNG key have a global state
# this is NOT in accordance with jax's philosophy on PRNGs, but
class _JKey:
    def __init__(self):
        self.jkey = None
        self.needs_reset = True
        pass
    
    def reset(self, seed: int):
        if self.needs_reset:
            self.jkey = jax.random.PRNGKey(seed)
            print('reset with seed {}'.format(seed))
            self.needs_reset = False
        pass
        
    def __call__(self):
        assert self.jkey is not None, 'must call `set_seed()` before using PRNG'
        self.jkey, key = jax.random.split(self.jkey)
        self.needs_reset = True
        return key
    
jkey = _JKey()
_seed = np.random.randint(10000)
print('WARNING: seed was randomly chosen to be {}'.format(_seed))

def set_seed(seed: int = None):
    if seed is None: 
        seed = _seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    jkey.reset(seed)
    pass
# --------------------------------------------------------------------------------
    
def sample(jkey: jax.random.KeyArray, 
           shape,
           sampling_method=SAMPLING_METHOD) -> jnp.ndarray:
    assert sampling_method in ['ball', 'sphere', 'rademacher']
    if sampling_method == 'ball':
        v = jax.random.ball(jkey, np.prod(shape), dtype=jnp.float32).reshape(*shape)
    elif sampling_method == 'sphere':
        v = jax.random.normal(jkey, shape, dtype=jnp.float32)
        v = v / jnp.linalg.norm(v)
    elif sampling_method == 'rademacher':
        v = jax.random.rademacher(jkey, shape, dtype=jnp.float32)
    assert v.shape == shape, 'needed {} and got {}'.format(shape, v.shape)
    return v
    
def _sigmoid(t):
    return 1 / (1 + np.exp(-t))

def _inv_sigmoid(s):
    return np.log(s / (1 - s + 1e-8))

def _d_sigmoid(t):
    s = _sigmoid(t)
    return s * (1 - s)

def rescale(t, bounds, use_sigmoid):
    """
    rescales from `[0, 1] -> [tmin, tmax]`
    """
    tmin, tmax = bounds
    if use_sigmoid: t = _sigmoid(t)
    return tmin + (tmax - tmin) * t

def d_rescale(t, bounds, use_sigmoid):
    tmin, tmax = bounds
    d = tmax - tmin
    if use_sigmoid: d *= _d_sigmoid(t)
    return d

def inv_rescale(s, bounds, use_sigmoid):
    tmin, tmax = bounds
    t = (s - tmin) / (tmax - tmin)
    if use_sigmoid: t = _inv_sigmoid(t)
    return t

def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""
    base = np.exp(np.log(end / start) / (num - 1))
    return [int(np.round(start * base**i / divisible_by) * divisible_by) for i in range(num)]

def append(arr, val):
    """
    rightmost recent appending, i.e. arr = (val_{t-h}, ..., val_{t-1}, val_t)
    """
    if isinstance(arr, jnp.ndarray):
        if not isinstance(val, jnp.ndarray):
            val = jnp.array(val, dtype=arr.dtype)
        arr = arr.at[0].set(val)
        arr = jnp.roll(arr, -1, axis=0)
    return arr

def window_average(seq, window_size: int, use_median: bool=False):
    ret = []
    for i in range(len(seq)):
        l = max(0, i - window_size // 2)
        r = min(i + window_size // 2, len(seq) - 1)
        ret.append(np.mean(seq[l: r]) if not use_median else np.median(seq[l: r]))
    return np.array(ret)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def top_1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        num_incorrect = (torch.softmax(logits, dim=-1).argmax(dim=-1) != targets).cpu().sum().item()
        return num_incorrect / logits.shape[0]

def V_pert(m,n):
    """ Form the V_{m,n} perturbation matrix as defined in the paper

    Args:
        m
        n

    Returns:
        V_{m,n}
    """
    V = torch.zeros((m*n,m*n))
    for i in range(m*n):
        block = ((i*m) - ((i*m) % (m*n))) / (m*n)
        col = (i*m) % (m*n)
        V[i,col + round(block)] = 1
    return V

def vec(A):
    """returns vec(A) of matrix A (i.e. columns stacked into a vector)

    Args:
        A

    Returns:
        vec(A)
    """
    m, n = A.shape
    vecA = torch.zeros((m*n, 1))
    for i in range(n):
        vecA[i*m:(i+1)*m,:] = A[:,i].unsqueeze(1)

    return vecA #torch.reshape(A, (m * n, 1)) #A.view(m*n, 1) # vecA

def inv_vec(v,A):
    """Inverse operation of vecA"""
    v_out = torch.zeros_like(A)
    m, n = A.shape
    for i in range(n):
        v_out[:,i] = v[0,i*m:(i+1)*m]
        
    return v_out #torch.reshape(v, (m, n)).T #v.view(m, n).T #v_out

def kronecker(A, B):
    """Kronecker product of matrices A and B"""
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

class Riccati(torch.autograd.Function):
    """
    lovingly borrowed from https://github.com/sebastian-east/dare-torch/blob/main/riccati.py 
    """

    @staticmethod                       #FORWARDS PASS
    def forward(ctx, A, B, Q, R):
        
        if not (A.type() == B.type() and A.type() == Q.type() and A.type() == R.type()):
            raise Exception('A, B, Q, and R must be of the same type.')
            
        Atemp = A.detach().numpy()
        Btemp = B.detach().numpy()
        Q = 0.5 * (Q + Q.transpose(0, 1))
        Qtemp = Q.detach().numpy()
        R = 0.5 * (R + R.transpose(0,1))
        Rtemp = R.detach().numpy()

        P = solve_discrete_are(Atemp, Btemp, Qtemp, Rtemp)
        P = torch.from_numpy(P).type(A.type())

        ctx.save_for_backward(P, A, B, Q, R) #Save variables for backwards pass
        return P

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = vec(grad_output).transpose(0,1).float()
        P, A, B, Q, R = ctx.saved_tensors
        n, m = B.shape
        #Computes derivatives using method detailed in paper
        
        M3 = R + B.transpose(0,1) @ P @ B
        M2 = M3.inverse()
        M1 = P - P @ B @ M2 @ B.transpose(0,1) @ P

        LHS = kronecker(B.transpose(0,1), B.transpose(0,1))
        LHS = kronecker(M2, M2) @ LHS
        LHS = kronecker(P @ B, P @ B) @ LHS
        LHS = LHS - kronecker(torch.eye(n), P@B@M2@B.transpose(0,1))
        LHS = LHS - kronecker(P @ B @ M2 @ B.transpose(0,1), torch.eye(n))
        LHS = LHS + torch.eye(n ** 2)
        LHS = kronecker(A.transpose(0,1), A.transpose(0,1)) @ LHS
        LHS = torch.eye(n ** 2) - LHS
        invLHS = torch.inverse(LHS)

        RHS = V_pert(n,n).type(A.type()) + torch.eye(n ** 2)
        RHS = RHS @ kronecker(torch.eye(n), A.transpose(0,1) @ M1)
        dA = invLHS @ RHS
        dA = grad_output @ dA
        dA = inv_vec(dA, A)

        RHS = kronecker(torch.eye(m), B.transpose(0,1) @ P)
        RHS = (torch.eye(m ** 2) + V_pert(m,m).type(A.type())) @ RHS
        RHS = -kronecker(M2, M2) @ RHS
        RHS = -kronecker(P@B, P@B) @ RHS
        RHS = RHS - (torch.eye(n ** 2) + V_pert(n,n).type(A.type())) @ (kronecker(P @ B @ M2, P))
        RHS = kronecker(A.transpose(0,1), A.transpose(0,1)) @ RHS
        dB = invLHS @ RHS                                                                             
        dB = grad_output @ dB
        dB = inv_vec(dB, B)

        RHS = torch.eye(n ** 2).float()
        dQ = invLHS @ RHS
        dQ = grad_output @ dQ
        dQ = inv_vec(dQ, Q)
        dQ = 0.5 * (dQ + dQ.transpose(0, 1))

        RHS = -kronecker(M2, M2)
        RHS = - kronecker(P @ B, P @ B) @ RHS
        RHS = kronecker(A.transpose(0,1), A.transpose(0,1)) @ RHS
        dR = invLHS @ RHS
        dR = grad_output @ dR
        dR = inv_vec(dR, R)
        dR = 0.5 * (dR + dR.transpose(0, 1))

        return dA, dB, dQ, dR
