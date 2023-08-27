import logging
from typing import List
import random
import scipy.optimize as optimize
from scipy.linalg import solve_discrete_are, orth
import math

import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn

import gymnasium as gym

# for rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation

COLORS = {
          # baselines for LQR
          'LQR': 'b',
          'HINF': 'purple',
          'GPC': 'orange',
          'BPC': 'g',
          'RBPC': 'r',
          
          # general baselines
          'Zero': 'b',
          'Constant': 'purple',
          
          # our methods
          'No Lift': 'm',
          'Random Lift': 'k',
          'Learned Lift': 'c',
          'Linear': 'brown',
          'Lifted': 'm'}

SAMPLING_METHOD = 'normal'  # must be in `['ball', 'sphere', 'rademacher', 'normal']``
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def render(experiment, xkey, ykey, sliderkey: str = None, save_path: str = None, duration: float = 5, fps: int = 30):
    assert experiment.stats is not None, 'cannot plot the results of an experiment that hasnt been run'
    all_stats = experiment.stats
    
    logging.info('(RENDERER): rendering the stats of the following: {}'.format(list(all_stats.keys())))
    ncols = 2 if sliderkey is not None else 1
    fig, ax = plt.subplots(1, ncols, figsize=(3 * ncols, 6))
    if sliderkey is not None:
        slider = (ax[1], sliderkey)
        ax = ax[0]
    else: slider = None
    init_fns = []
    render_fns = []
    num_frames = duration * fps
    for k, s in all_stats.items():
        init_fn, render_fn = s.get_render_fns(ax, xkey, ykey, label=k, slider=slider, num_frames=num_frames)
        init_fns.append(init_fn)
        render_fns.append(render_fn)
    def init_render():
        objs = []
        for fn in init_fns: objs.extend(fn())
        return objs
    def render(i):
        objs = []
        for fn in render_fns: objs.extend(fn(i))
        return objs
    _ax = ax; _ax.legend(); _ax.set_xlabel(xkey); _ax.set_ylabel(ykey)
    if sliderkey is not None: _ax = slider[0]; _ax.legend(); _ax.set_xlabel('timestep (t)'); _ax.set_ylabel(sliderkey)
    anim = animation.FuncAnimation(fig, render, init_func=init_render, frames=num_frames, interval=1000 / fps)
    if save_path is not None:
        anim.save(save_path, writer=animation.FFMpegWriter(fps=fps))  # saving to mp4 using ffmpeg writer
        logging.info('(RENDERER): saved rendering to `{}`'.format(save_path))
    plt.close(fig)
    return anim

def ylim(ax, left, right):
    _l, _r = ax.get_ylim()
    l = max(_l, left)
    r = min(_r, right)
    ax.set_ylim(l, r)
    return ax

def get_color(method: str):
    if method is None or method not in COLORS:
        logging.warning('(UTILS): no hardcoded color for `{}`. using a random one :)'.format(method))
        color = None
    else:
        color = COLORS[method]
    return color

def rescale_ax(ax, x, y, margin = 0.08, ignore_current_lims: bool = False):
    if ignore_current_lims:
        xl, xr = x, x
        yl, yr = y, y
    else:
        xl, xr = ax.get_xlim()
        yl, yr = ax.get_ylim()
    xh, yh = (xl + xr) / 2, (yl + yr) / 2
    xv = margin * abs(x - xh)
    yv = margin * abs(y - yh)
    xl, xr = min(xl, x - xv), max(xr, x + xv)
    yl, yr = min(yl, y - yv), max(yr, y + yv)
    ax.set_xlim(xl, xr + 1e-8)
    ax.set_ylim(yl, yr + 1e-8)
    return ax
    

def get_classname(obj):
    return str(obj.__class__).split('.')[-1].upper()[:-2]

def summarize_lds(A, B):
    s = ''
    s += '||A||_op = {}'.format(opnorm(A))
    s += '\n||B||_F = {}'.format(jnp.linalg.norm(B, ord='fro'))
    try:
        s += '\n||A-BK||_op = {}'.format(opnorm(A - B @ dare_gain(A, B)))
    except:
        s += '\n||A-BK||_op folded (couldnt find finite solution)'
    s += '\neig(A) = {}'.format(np.sort([jnp.linalg.norm(e) for e in jnp.linalg.eigvals(A)])[::-1])
    s += '\nsvd(B) = {}'.format(jnp.linalg.svd(B)[1])
    return s

def opnorm(X):
    if isinstance(X, torch.Tensor):
        return torch.amax(torch.abs(torch.linalg.svd(X).S))
    elif isinstance(X, np.ndarray):
        return np.amax(np.abs(np.linalg.svd(X)[1]))
    elif isinstance(X, jnp.ndarray):
        return jnp.amax(jnp.abs(jnp.linalg.svd(X)[1])).item()
    else:
        raise NotImplementedError(X.__class__)
    
def dare_gain(A, B, Q = None, R = None):    
    if isinstance(A, torch.Tensor):
        if Q is None: Q = torch.eye(B.shape[0])
        if R is None: R = torch.eye(B.shape[1])
        P = Riccati.apply(A, B, Q, R)
        K = torch.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # compute LQR gain
    elif isinstance(A, np.ndarray):
        if Q is None: Q = np.eye(B.shape[0])
        if R is None: R = np.eye(B.shape[1])
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # compute LQR gain
    elif isinstance(A, jnp.ndarray):
        if Q is None: Q = jnp.eye(B.shape[0])
        if R is None: R = jnp.eye(B.shape[1])
        P = solve_discrete_are(A, B, Q, R)
        K = jnp.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)  # compute LQR gain
    return K

def method_of_moments(xs: jnp.ndarray, 
                      us: jnp.ndarray,
                      mask: jnp.ndarray = None):
    """
    runs method of moments to find A, B s.t.
        `A @ x_{t} + B @ u_{t} = x_{t+1}`
    """
    T0 = xs.shape[0]
    k = max(int(0.01 * T0), 5)
    scan_len = T0 - k - 1
    
    if isinstance(xs, torch.Tensor):
        # prepare vectors and retrieve B
        N_j = torch.stack([xs[j + 1: j + scan_len + 1].T @ us[:scan_len] for j in range(1, k + 1)], dim=0) / scan_len
        B = N_j[0]
        
        # retrieve A
        C_0, C_1 = N_j[:-1], N_j[1:]
        C_inv = torch.linalg.inv(torch.tensordot(C_0, C_0, dims=((0, 2), (0, 2))) + 1e-6 * torch.eye(xs.shape[1]))
        A = torch.tensordot(C_1, C_0, dims=((0, 2), (0, 2))) @ C_inv
        return A, B
        
    elif isinstance(xs, np.ndarray):
        xs, us = jnp.array(xs), jnp.array(us)
        
    # prepare vectors and retrieve B
    N_j = jnp.array([jnp.dot(xs[j + 1: j + scan_len + 1].T, us[:scan_len]) for j in range(k + 1)]) / scan_len
    B = N_j[0] # jnp.dot(states[1:].T, eps[:-1]) / (self.t - 1)

    # retrieve A
    C_0, C_1 = N_j[:-1], N_j[1:]
    C_inv = jnp.linalg.inv(jnp.tensordot(C_0, C_0, axes=((0, 2), (0, 2))) + 1e-6 * jnp.identity(xs.shape[1]))
    A = jnp.tensordot(C_1, C_0, axes=((0, 2), (0, 2))) @ C_inv
    
    return A, B
    
def least_squares(xs: jnp.ndarray, 
                  us: jnp.ndarray, 
                  mask: jnp.ndarray = None,
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
    
    if mask is None:
        if isinstance(xs, torch.Tensor): mask = torch.ones(xs.shape[0] - 1, dtype=bool)
        elif isinstance(xs, (np.ndarray, jnp.ndarray)): mask = np.ones(xs.shape[0] - 1, dtype=bool) 
        else: raise NotImplementedError(xs.__class__)
         
    if isinstance(xs, torch.Tensor):
        assert isinstance(us, torch.Tensor), us.__class__
        ret = torch.linalg.lstsq(torch.hstack((x_in, u_in))[mask], x_out[mask], rcond=-1).solution
        A, B = ret[:ds].T, ret[ds:].T
        return A, B
    
    if max_opnorm is None:
        A_B = np.linalg.lstsq(np.hstack((x_in, u_in))[mask], x_out[mask], rcond=-1)[0]
        A, B = A_B[:ds].T, A_B[ds:].T
    else:
        # logging.info('(UTILS) constraining operator norm of `A` to be <= {}'.format(max_opnorm))
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
        # if self.needs_reset:
        self.jkey = jax.random.PRNGKey(seed)
        # print('reset with seed {}'.format(seed))
        self.needs_reset = False
        pass
        
    def __call__(self):
        assert self.jkey is not None, 'must call `set_seed()` before using PRNG'
        self.jkey, key = jax.random.split(self.jkey)
        self.needs_reset = True
        return key
    
jkey = _JKey()
random_numbers = list(np.random.randint(10000, size=(10000,)))

def set_seed(seed: int = None, meta_seed: int = None):
    global random_numbers
    if meta_seed is not None or len(random_numbers) == 1:
        if meta_seed is not None:
            np.random.seed(meta_seed)
            logging.debug('(UTILS) set meta seed to {}'.format(meta_seed))
        random_numbers = list(np.random.randint(10000, size=(10000,)))
    if seed is None:
        seed = random_numbers.pop()
    logging.debug('(UTILS) set seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    jkey.reset(seed)
    pass
# --------------------------------------------------------------------------------
    
def sample(jkey: jax.random.KeyArray, 
           shape,
           sampling_method=SAMPLING_METHOD) -> jnp.ndarray:
    assert sampling_method in ['ball', 'sphere', 'rademacher', 'normal']
    assert len(shape) > 0
    if sampling_method == 'ball':
        if len(shape) == 1: v = jax.random.ball(jkey, d=shape[0], dtype=jnp.float32).reshape(*shape)
        else: v = jax.random.ball(jkey, np.prod(shape[1:]), shape=shape[:1], dtype=jnp.float32).reshape(*shape)  # if dim >= 2, assumes batch and flatten rest
    elif sampling_method == 'sphere':
        assert len(shape) <= 2, 'expected either 1D vector or 2D with a batch dim'
        v = jax.random.normal(jkey, shape, dtype=jnp.float32)
        v = v / jnp.linalg.norm(v, axis=-1).reshape(*shape[:-1], 1)
    elif sampling_method == 'rademacher':
        v = jax.random.rademacher(jkey, shape, dtype=jnp.float32)
    elif sampling_method == 'normal':
        v = jax.random.normal(jkey, shape, dtype=jnp.float32)
    assert v.shape == shape, 'needed {} and got {}'.format(shape, v.shape)
    return v
    
def _arctanh(t):
    """
    returns inverse of tanh
    """ 
    assert jnp.all(jnp.abs(t) <= 1), 'must be in `[-1, 1]^n` to take inverse hyperbolic tan'   
    return 0.5 * jnp.log((1 + t) / (1 - t + 1e-8))

def rescale(t, bounds, use_tanh) -> jnp.ndarray:
    """
    rescales from `[-1, 1] -> [tmin, tmax]`
    
        `s = (0.5 + clip(t, -1, 1) / 2) * (tmax - tmin) + tmin
    """
    tmin, tmax = bounds
    t = jnp.tanh(t) if use_tanh else jnp.clip(t, -1, 1)
    t = (1 + t) / 2
    t = (tmax - tmin) * t + tmin
    return t

def d_rescale(t, bounds, use_tanh) -> jnp.ndarray:
    tmin, tmax = bounds
    d = (tmax - tmin) / 2
    if use_tanh: d *= 1 - jnp.tanh(t) ** 2
    return d

def inv_rescale(s, bounds, use_tanh) -> jnp.ndarray:
    tmin, tmax = bounds
    t = (s - tmin) / (tmax - tmin)
    t = jnp.clip(2 * t - 1, -1, 1)
    if use_tanh: 
        t = _arctanh(t)
    return t

# Method to create sinusoidal timestep embeddings, much like positional encodings found in many Transformers
def timestep_embedding(timesteps, embedding_dim, method='sin', max_period=10000):
    """
    Embeds input timesteps

    Parameters
    ----------
    timesteps : jnp.ndarray
        input timesteps
    embedding_dim : int
        dimension for each scalar to embed to
    method : str, optional
        how to perform embedding, by default 'sin', must be one of `['sin', 'identity']`
    max_period : int, optional
        maximum period for sinusoidal embeddings, by default 10000
    """
    if not isinstance(timesteps, jnp.ndarray):
        timesteps = jnp.array(timesteps)
    if timesteps.ndim == 1: timesteps = timesteps[None]
    if method == 'sin':
        half = embedding_dim // 2
        emb = math.log(max_period) / half
        emb = jnp.exp(jnp.arange(half, dtype=float) * -emb)
        emb = timesteps[:, None].astype(float) * emb[None]
        emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=1)

        if embedding_dim % 2 == 1:  # Zero pad for odd dimensions, ty to https://stackoverflow.com/questions/69453600/inverse-operation-to-padding-in-jax for conversion to jax
            pad = (0, 1, 0, 0)
            value = 0.
            pad = list(zip(*[iter(pad)]*2))
            pad += [(0, 0)] * (emb.ndim - len(pad))
            emb = jax.lax.pad(emb, padding_config=[(i, j, 0) for i, j in pad[::-1]], padding_value=jnp.array(value, emb.dtype))
    elif method == 'identity':
        emb = jnp.expand_dims(timesteps, axis=-1).expand([*timesteps.shape, embedding_dim])
        logging.warning('(UTILS): using identity timestep embeddings. not too sure if this is a good idea')
    return emb.reshape(embedding_dim)

def exponential_linspace_int(start, end, num, divisible_by=1) -> List[int]:
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


"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Continuous version by Ian Danforth
"""

from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}, False

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state), {}

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            raise NotImplementedError('couldnt import rendering')
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()

def random_lds(
    state_dim: int, 
    control_dim: int,
    obs_dim: int = None, 
    p_int_first=0.1, p_int_others=0.01, p_repeat=0.05, p_complex=0.5
):
    """
    !!!! I YANKED THIS FROM pykoopman !!!!!
    uses numpy's PRNG
    
    Create a discrete-time, random, stable, linear state space model.

    Args:
        n (int, optional): Number of states. Default is 2.
        p (int, optional): Number of control inputs. Default is 2.
        m (int, optional): Number of output measurements.
            If m=0, C becomes the identity matrix, so that y=x. Default is 2.
        p_int_first (float, optional): Probability of an integrator as the first pole.
            Default is 0.1.
        p_int_others (float, optional): Probability of other integrators beyond the
            first. Default is 0.01.
        p_repeat (float, optional): Probability of repeated roots. Default is 0.05.
        p_complex (float, optional): Probability of complex roots. Default is 0.5.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple containing the
        state transition matrix (A), control matrix (B), and measurement matrix (C).

        A (numpy.ndarray): State transition matrix of shape (n, n).
        B (numpy.ndarray): Control matrix of shape (n, p).
        C (numpy.ndarray): Measurement matrix of shape (m, n). If m = 0, C is the
            identity matrix.

    """
    n, p, m = state_dim, control_dim, obs_dim
    if obs_dim is None: m = state_dim

    # Number of integrators
    nint = int(
        (np.random.rand(1) < p_int_first) + sum(np.random.rand(n - 1) < p_int_others)
    )
    # Number of repeated roots
    nrepeated = int(np.floor(sum(np.random.rand(n - nint) < p_repeat) / 2))
    # Number of complex roots
    ncomplex = int(
        np.floor(sum(np.random.rand(n - nint - 2 * nrepeated, 1) < p_complex) / 2)
    )
    nreal = n - nint - 2 * nrepeated - 2 * ncomplex

    # Random poles
    rep = 2 * np.random.rand(nrepeated) - 1
    if ncomplex != 0:
        mag = np.random.rand(ncomplex)
        cplx = np.zeros(ncomplex, dtype=complex)
        for i in range(ncomplex):
            cplx[i] = mag[i] * np.exp(complex(0, np.pi * np.random.rand(1)))
        re = np.real(cplx)
        im = np.imag(cplx)

    # Generate random state space model
    A = np.zeros((n, n))
    if ncomplex != 0:
        for i in range(0, ncomplex):
            A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.array(
                [[re[i], im[i]], [-im[i], re[i]]]
            )

    if 2 * ncomplex < n:
        list_poles = []
        if nint:
            list_poles = np.append(list_poles, np.ones(nint))
        if any(rep):
            list_poles = np.append(list_poles, rep)
            list_poles = np.append(list_poles, rep)
        if nreal:
            list_poles = np.append(list_poles, 2 * np.random.rand(nreal) - 1)

        A[2 * ncomplex :, 2 * ncomplex :] = np.diag(list_poles)

    T = orth(np.random.rand(n, n))
    A = np.transpose(T) @ (A @ T)

    # control matrix
    B = np.random.randn(n, p)
    # mask for nonzero entries in B
    mask = np.random.rand(B.shape[0], B.shape[1])
    B = np.squeeze(np.multiply(B, [(mask < 0.75) != 0]))

    # Measurement matrix
    if m == 0:
        C = np.identity(n)
    else:
        C = np.random.randn(m, n)
        mask = np.random.rand(C.shape[0], C.shape[1])
        C = np.squeeze(C * [(mask < 0.75) != 0])

    return (A, B, C) if obs_dim is not None else (A, B)
