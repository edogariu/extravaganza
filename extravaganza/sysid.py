import logging
from abc import abstractmethod
from collections import defaultdict
from typing import List, Callable, Dict
from copy import deepcopy
import tqdm

import numpy as np
import jax
import jax.numpy as jnp
import torch
import pykoopman as pk

from extravaganza.controllers import Controller
from extravaganza.observables import Trajectory
from extravaganza.models import TorchMLP, TorchLifter
from extravaganza.stats import Stats
from extravaganza.utils import set_seed, jkey, sample, get_classname, least_squares, method_of_moments, summarize_lds, opnorm

KOOPMAN_METHODS = ['polynomial', 'rbf', 'fourier']
MAX_OPNORM = None  # set this to `None` to have unconstrained opnorm of `A` in our regressions
LOSS_WEIGHTS = {
    # touch these
    'isometry': 0,
    'nontrivial': 1e-3,
    'jac': 0,
    'dot': 0,
    'vmf': 0,
    'jac_conditioning': 0,
    'l2 linearization': 1,
    'l2 linearization relative': 0,
    'l1 linearization': 0,
    'reconstruction': 0,
    'injectivity': 0,
    'surjectivity': 0,
    'cpc': 0,
    'guess the control': 0,
    'simplification': 1,
    'residual centeredness': 0,
    'centeredness': 0,
    
    # dont touch these
    'consistency': 0.1,
    'proj_isometry': 1,
    'proj_error': 1,
}
GET_LATENT_DIM = lambda state_dim: state_dim
RESCALE_AFTER_TRAINING = False

GEN_LENGTH = 5000
DO_RESET_MASK_DEBUG = False  # to confirm whether the reset masks are correct. can only be used with CartPole

class Explorer:
    def __init__(self, seq_args: Dict[str, Dict], initial_control: jnp.ndarray):
        self.control_dim = initial_control.shape[0]
        self.initial_control = initial_control
        self.t = 0
        
        keys, probs, args  = [], [], {}
        for k, v in seq_args.items():
            t = k.split(' ')
            assert len(t) == 2, 'improper exploration key format. should be something like `random 0.8`'
            keys.append(t[0]); probs.append(float(t[1])); args[t[0]] = v
        probs = jnp.array(np.array(probs))
        
        assert jnp.allclose(probs.sum(), 1.), 'exploration sequence probabilities dont sum to 1?'
        assert jnp.allclose(jax.nn.softmax(jnp.log(probs)), probs); probs = jnp.log(probs)
        
        seq_fns = {'random': (self._prep_random, self._sample_random),
                   'repeat': (self._prep_repeat, self._sample_repeat),
                   'impulse': (self._prep_impulse, self._sample_impulse)}
        for k, (prep_fn, _) in seq_fns.items():
            if k in keys: 
                args[k] = prep_fn(args[k])
                if 'scales' in args[k] and not hasattr(self, 'exploration_scales'): self.exploration_scales = args[k]['scales']
        for k in keys: assert k in seq_fns, '{} is an unknown key!'.format(k)
        
        def sample_single_sequence():
            key = keys[jax.random.categorical(jkey(), probs).item()]
            return seq_fns[key][1](args[key])
        
        def generate_exploration_controls(length: int):
            logging.info('(EXPLORER) generating exploration control sequences using {} w.p. {}'.format(keys, jax.nn.softmax(probs)))
            controls = []
            pbar = tqdm.tqdm(total=length)
            while len(controls) < length: 
                seq = sample_single_sequence()
                controls.extend(seq)
                pbar.update(len(seq))
            pbar.close(); del pbar
            controls = controls[:length]
            return controls
        self.GEN_LENGTH = GEN_LENGTH
        self.generate_exploration_controls = generate_exploration_controls
        self.controls = self.generate_exploration_controls(self.GEN_LENGTH)
        pass
    
    def explore(self):
        if self.t >= len(self.controls):
            self.controls = self.generate_exploration_controls(self.GEN_LENGTH)
            self.t = 0
        control = self.controls[self.t]
        self.t += 1
        return control
    
    # ------------------- DICTIONARY OF SEQUENCES FOR EXPLORATION --------------------------

    def _prep_random(self, args: Dict):  # prepares the args of dict (inplace!) to be used in the corresponding sample method
        scales, bounds = args['scales'], args['bounds'] if 'bounds' in args else None
        if isinstance(scales, float): scales = [scales for _ in range(self.control_dim)]
        scales = jnp.array(np.array(scales))
        assert scales.shape == (self.control_dim,)
        
        if bounds is not None:
            if isinstance(bounds[0], float): 
                assert len(bounds) == 2
                bounds = [bounds for _ in range(self.control_dim)]
            bounds = jnp.array(np.array(bounds))
            assert bounds.shape == (self.control_dim, 2)
        args['scales'], args['bounds'] = scales, bounds
        return args

    def _sample_random(self, args: Dict):
        scales, bounds = args['scales'], args['bounds']
        control = scales * sample(jkey(), shape=(self.control_dim,)) + self.initial_control
        if bounds is not None: 
            for i in range(self.control_dim): control = control.at[i].set(jnp.clip(control[i], *bounds[i]))
        return [control]
    
    def _prep_repeat(self, args: Dict):
        args = self._prep_random(args)
        assert args['avg_len'] > 2
        return args
    
    def _sample_repeat(self, args: Dict):
        avg_len = args['avg_len']
        repeat = jax.random.randint(jkey(), shape=(), minval=avg_len - avg_len // 3, maxval=avg_len + avg_len // 3 + 1)
        control = self._sample_random(args)[0]
        return [control for _ in range(repeat)]
    
    def _prep_impulse(self, args: Dict):
        args = self._prep_repeat(args)
        return args
    
    def _sample_impulse(self, args: Dict):
        avg_len = args['avg_len']
        zeros = [jnp.zeros(self.control_dim) for _ in range(jax.random.randint(jkey(), shape=(), minval=avg_len - avg_len // 3, maxval=avg_len + avg_len // 3 + 1) - 1)]
        control = self._sample_random(args)[0]
        return [control, *zeros]
    
    # -------------------------------------------------------------------------------------------------------------


class SystemModel:
    def __init__(self,
                 obs_dim: int,
                 control_dim: int,
                 state_dim: int,
                 
                 exploration_args: Dict[str, Dict],  # which type of exploration sequence. key SHOULD BE SEQ NAME, then a SPACE, then the PROBABILITY
                 
                 initial_control: jnp.ndarray = None,
                 stats: Stats = None,
                 seed: int = None):

        set_seed(seed)
        
        self.obs_dim: int = obs_dim
        self.control_dim: int = control_dim
        self.state_dim: int = state_dim        
        
        self.trajs: List[Trajectory] = [Trajectory()]
        
        if initial_control is None: initial_control = jnp.zeros(self.control_dim)
        assert initial_control.shape == (self.control_dim,)
        self.explorer = Explorer(exploration_args, initial_control)
        
        for d in [obs_dim, control_dim, state_dim]: assert d > 0
        
        if DO_RESET_MASK_DEBUG:
            logging.info('(RESET MASK DEBUG) testing to see if the reset masks are correct')
            assert self.control_dim == 1 and self.obs_dim == 4, '(RESET MASK DEBUG) cant do the test if ur not in cartpole' 
        
        # stats to keep track of
        self.trained = False
        self.t = 0
        self.stats = stats
        if self.stats is None:
            logging.debug('({}): no `Stats` object provided, so a new one will be made.'.format(get_classname(self)))
            self.stats = Stats()
        pass
            
    
    def explore(self, cost: float, obs: jnp.ndarray):  # cost for entering into `obs`, where we are right now
        assert obs.shape == (self.obs_dim,), obs.shape

        # `control` is always the control played AFTER receiving `obs`
        self.trajs[-1].add_state(cost, obs)
        control = self.explorer.explore()
        if DO_RESET_MASK_DEBUG: control = jnp.array([0.2])  # always positive in order to do the debug
        self.trajs[-1].add_control(control)

        self.t += 1
        return control
    
    def end_trajectory(self):
        if len(self.trajs[-1]) > 0: self.trajs.append(Trajectory())
        pass
    
    def concatenate_trajectories(self):
        x, u, f = deepcopy(self.trajs[0].x), deepcopy(self.trajs[0].u), deepcopy(self.trajs[0].f)
        for traj in self.trajs[1:]:
            x.extend(traj.x)
            u.extend(traj.u)
            f.extend(traj.f)
        x, u, f = map(lambda arr: jnp.stack(arr, axis=0), [x, u, f])
        return x, u, f
    
    @abstractmethod
    def end_exploration(self):
        """
        this is where the system model makes use of the exploration dataset to train and do whatever
        """
        self.trained = True
        pass
    
    @abstractmethod
    def get_state(self, 
                  obs: jnp.ndarray,
                  sq_norm: float = None): 
        """
        should be identity operation if we are not lifting
        """
        pass
    

class Lifter(SystemModel):
    def __init__(self,
                 obs_dim: int,
                 control_dim: int,
                 state_dim: int,
                 
                 exploration_args: Dict[str, Dict],  # which type of exploration sequence. key SHOULD BE SEQ NAME, then a SPACE, then the PROBABILITY
                 method: str,   # must be in ['identity', 'polynomial', rbf', 'fourier', 'nn']
                  
                 AB_method: str,
                 depth: int = 4,  # depth of NN
                 sigma: float = 0.,
                 deterministic: bool = True,
                 isometric: bool = False,
                 
                 num_iters: int = 8000,
                 batch_size: int = -1,
                 lifter_lr: float = 0.001,
                 
                 hh: int = 1,  # how many data points after a reset to ignore
                 initial_control: jnp.ndarray = None,
                 stats: Stats = None,
                 seed: int = None):
        
        super().__init__(obs_dim=obs_dim, control_dim=control_dim, state_dim=state_dim, 
                         exploration_args=exploration_args, initial_control=initial_control, stats=stats, seed=seed)
        self.method = method
        self.hh = hh
        
        # to compute lifted states which hopefully respond linearly to the controls
        if method == 'identity':
            assert self.obs_dim == self.state_dim
        
        elif method in KOOPMAN_METHODS:
            if method == 'fourier': assert (self.state_dim - self.obs_dim) % 2 == 0, '`state_dim - obs_dim` must be even (not {}) for Koopman fourier methods!'.format(self.state_dim - self.obs_dim)
            if method == 'polynomial': raise NotImplementedError('oopsie, for some reason polynomial observations isnt working yet')

            observables = {
                'polynomial': pk.observables.Polynomial(degree=3),
                'rbf': pk.observables.RadialBasisFunction(rbf_type='gauss', n_centers=self.state_dim - self.obs_dim, include_state=True),
                'fourier': pk.observables.RandomFourierFeatures(D=(self.state_dim - self.obs_dim) // 2, include_state=True),
            }
            self.model = pk.Koopman(observables=observables[method], regressor=pk.regression.EDMDc())

        elif method == 'nn':
            self.num_iters = num_iters
            self.batch_size = batch_size
            
            for i in range(self.hh): 
                for k in ['l2 linearization', 'l2 linearization relative', 'l1 linearization', 'simplification']:
                    LOSS_WEIGHTS[k + ' ' + str(i)] = LOSS_WEIGHTS[k]
            
            latent_dim = GET_LATENT_DIM(self.state_dim)  # always be linearizing in a higher dimension
            layer_dims = [self.obs_dim, *[5 * latent_dim for _ in range(depth - 1)], latent_dim]
            enc = TorchMLP(layer_dims,
                           activation=torch.nn.Tanh,
                        #    normalization=torch.nn.LayerNorm,
                           drop_last_activation=True,
                           use_bias=True,
                           seed=seed)
            
            self.lifter = TorchLifter(enc, self.obs_dim, self.control_dim, self.state_dim, latent_dim,
                                      hh=self.hh, isometric=isometric, deterministic=deterministic,
                                      AB_method=AB_method, sigma=sigma, loss_weights=LOSS_WEIGHTS).float()
            self.lifter_opt = torch.optim.AdamW(self.lifter.parameters(), lr=lifter_lr, weight_decay=1e-3)

        else: raise NotImplementedError(method)
        
        self.trained = False
        pass
    
    def dynamics(self, z, u): return (self.A @ z.unsqueeze(-1) + self.B @ u.unsqueeze(-1)).squeeze(-1)
    
    def end_exploration(self, wordy: bool = True):
        logging.info('({}): ending sysid phase at step {}'.format(get_classname(self), self.t))        
        states, controls, costs = self.concatenate_trajectories()
        assert states.shape[0] == controls.shape[0] and states.shape[0] == costs.shape[0]
        
        # find the index of the last datapoint in each trajectory. we will ignore loss terms referencing x_{t-hh}, ..., x_{t+hh} (inclusive) for all t in ignore_idxs,
        # since a reset happened between those two points
        traj_lens = [len(traj) for traj in self.trajs]
        ignore_idxs = np.cumsum(traj_lens) - 1  # -1 to capture last idxs of old trajs, not first idxs of new trajs
        mask = np.ones(states.shape[0], dtype=bool)
        for idx in ignore_idxs: mask[max(0, idx - self.hh + 1): idx + self.hh + 1] = False  # anything within +/- hh of a reset must go
        mask[:self.hh] = False  # the first hh must go
        mask = mask[:-1]
        
        if DO_RESET_MASK_DEBUG:
            # -------------------------------------------------------------------------------
            # ------- anomaly detection via cartpole to see if the masks work ---------------
            # -------------------------------------------------------------------------------
            delta_v = states[1:, 1] - states[:-1, 1]  # change in horizontal cart velocity
            diff = jnp.abs(jnp.sign(delta_v) - jnp.sign(controls[:-1].squeeze(-1))) == 0  # where the signs agree; they will disagree if and only if it was a reset
            assert jnp.all(diff == mask), '(RESET MASK DEBUG) failed :('
            logging.info('(RESET MASK DEBUG) passed :)')
        
        # -------------------------------------------------------------------------------
        
        self.AB = {}  # for storing sysids from various different techniques
        
        if self.method == 'identity':
            self.AB['regression'] = least_squares(states, controls, mask=mask, max_opnorm=MAX_OPNORM)
            self.AB['moments'] = method_of_moments(states, controls, mask=mask)
            ret = 'regression'
        
        elif self.method in KOOPMAN_METHODS:
            x, u = np.array(states), np.array(controls)
            self.model.fit(x=x, u=u)
            z = self.model.observables.transform(x.reshape(-1, self.obs_dim))
            
            self.AB['regression'] = least_squares(z, u, mask=mask, max_opnorm=MAX_OPNORM) 
            self.AB['moments'] = method_of_moments(z, u, mask=mask)           
            self.AB['koopman'] = (jnp.array(self.model.A.reshape(self.state_dim, self.state_dim)), jnp.array(self.model.B.reshape(self.state_dim, self.control_dim)))
            ret = 'regression'
            
        elif self.method == 'nn':  
                     
            # ----------------------------
            # DATASET
            # ----------------------------
            
            x, u, f = map(lambda arr: torch.tensor(np.array(arr)), [states, controls, costs])  # convert to tensors
            
            # normalize observations to be centered with unit variance. DONT TOUCH CONTROLS
            mean, std = torch.mean(x, dim=0), torch.std(x, dim=0)
            self.normalize_x = lambda t: (t - mean) / std
            
            # normalize costs to be nonnegative with mean 1, assuming they were nonnegative to begin with. DONT TOUCH CONTROLS
            fmean = torch.mean(f).item()
            if RESCALE_AFTER_TRAINING:
                self.sqrt_fmean = fmean ** 0.5
                logging.info('(LIFTER): NOTE THAT WE ARE USING FMEAN TO RESCALE EMBEDDINGS DURING INFERENCE (so outputs of `get_state()` will have their true norms instead of their normalized ones)!!!')
            else: self.sqrt_fmean = 1
            self.normalize_f = lambda t: t / fmean
            
            x = self.normalize_x(x)
            f = self.normalize_f(f)
            mask = torch.tensor(np.array(mask))
            X, U, F, MASK = x, u, f, mask
            if self.batch_size > 0: assert self.batch_size < X.shape[0]
            
            # # initialize our encoder to be the identity embedding
            # C = torch.zeros(self.state_dim, self.obs_dim)
            # for i in range(min(self.state_dim, self.obs_dim)): C[i, i] = 1.
            # for _ in range(2000):
            #     self.lifter_opt.zero_grad()
            #     z, _ = self.lifter.encode(x, f)
            #     loss = torch.nn.functional.mse_loss(z, (C @ x.unsqueeze(-1)).squeeze(-1))
            #     loss.backward()
            #     self.lifter_opt.step()
                
            # ----------------------------
            # TRAIN LOOP
            # ----------------------------
            
            print_every = max(self.num_iters // 10, 1)
            logging.info('training!')
            overall_losses = defaultdict(list)

            for i_iter in range(self.num_iters):
                
                if self.batch_size > 0:  # sample a 'batch', i.e. a slice of the training trajectory
                    idx = np.random.randint(X.shape[0] - self.batch_size - 1)
                    x, u, f, mask = (X[idx: idx + self.batch_size], 
                                     U[idx: idx + self.batch_size], 
                                     F[idx: idx + self.batch_size], 
                                     MASK[idx: idx + self.batch_size - 1])
                else: x, u, f, mask = X, U, F, MASK
                
                self.lifter_opt.zero_grad()
                
                (z, zhat), (A, B), losses = self.lifter.get_embs_and_losses(x, u, sq_norms=f, mask=mask)   
                zprev, zgt = z[:-1], z[1:]
                
                # how well the squared norm represents cost
                simp, l2, l1 = LOSS_WEIGHTS['simplification'], LOSS_WEIGHTS['l2 linearization'] or LOSS_WEIGHTS['l2 linearization relative'], LOSS_WEIGHTS['l1 linearization']
                if any((simp, l2, l2)): 
                    if simp: losses['simplification'] = torch.nn.functional.mse_loss(torch.norm(zprev[mask], dim=-1) ** 2, f[:-1][mask])
                    _z = z[:-self.hh]
                    for i in range(self.hh):
                        l, r = i, len(z) - self.hh + i
                        _m = mask[l:r]
                        _z = (A @ _z.unsqueeze(-1) + B @ u[l:r].unsqueeze(-1)).squeeze(-1)
                        if l2: 
                            if LOSS_WEIGHTS['l2 linearization']: losses[f'l2 linearization {i}'] = torch.nn.functional.mse_loss(_z[_m], z[l+1:r+1][_m])
                            if LOSS_WEIGHTS['l2 linearization relative']: losses[f'l2 linearization relative {i}'] = torch.mean((torch.norm(_z[_m] - z[l+1:r+1][_m], dim=-1) ** 2) / (torch.norm(z[l+1:r+1][_m], dim=-1) ** 2))
                        if l1: losses[f'l1 linearization {i}'] = torch.nn.functional.l1_loss(_z[_m], z[l+1:r+1][_m])
                        if simp: losses[f'simplification {i}'] = torch.nn.functional.mse_loss(torch.norm(_z[_m], dim=-1) ** 2, f[l+1:r+1][_m])
                
                if LOSS_WEIGHTS['dot']:  # compares empirical expectation to the theoretical one
                    diffs = (zgt - zprev)[mask]
                    diffs = (B.T @ diffs.unsqueeze(-1)).squeeze(-1)
                    RHS = (self.explorer.exploration_scales.item() * torch.norm(B, p='fro')) ** 2                    
                    LHS = (diffs * u[:-1][mask]).sum(dim=-1).mean()
                    losses['dot'] = torch.nn.functional.mse_loss(LHS, RHS)
                
                # how well we can reproduce the controls we used --   znext = A @ z + B @ u  ->  B^-1_left @ (znext - A @ z) = u
                if LOSS_WEIGHTS['guess the control']:
                    pinv = torch.linalg.pinv(B)
                    assert torch.allclose(pinv @ B, torch.eye(B.shape[1]), rtol=1e-1, atol=1e-1), (pinv @ B)
                    # pinv = torch.clamp(pinv, -1e2, 1e2)
                    uhat = (pinv @ (zgt.unsqueeze(-1) - (A @ zprev.unsqueeze(-1)))).squeeze(-1) 
                    losses['guess the control'] = torch.nn.functional.mse_loss(uhat[mask], u[:-1][mask])
                
                # injectivity proxy to penalize instances where two observations are distinct but are embedded similarly
                if LOSS_WEIGHTS['injectivity']:
                    _threshold = 1e-2
                    _x, _z = x[:-1][mask], zprev[mask]
                    # _z = _z / torch.norm(_z, dim=-1).mean()
                    obs_sq_dists = torch.triu(torch.cdist(_x.unsqueeze(0), _x.unsqueeze(0)).squeeze(0) ** 2, diagonal=1).reshape(-1)
                    emb_sq_dists = torch.triu(torch.cdist(_z.unsqueeze(0), _z.unsqueeze(0)).squeeze(0) ** 2, diagonal=1).reshape(-1)
                    _mask = obs_sq_dists > _threshold
                    # losses['injectivity'] = torch.corrcoef(torch.vstack((obs_sq_dists, emb_sq_dists)))[0, 1] ** 2 
                    # losses['injectivity'] = (1 / (emb_sq_dists + 1e-6)).mean()
                    losses['injectivity'] = (obs_sq_dists / (emb_sq_dists + 1e-6))[_mask].mean()
                    
                if LOSS_WEIGHTS['nontrivial']:
                    losses['nontrivial'] = 1 / (torch.norm(B, p='fro') ** 2 + 1e-6)
                
                # surjectivity proxy to ensure embeddings span their output space by ensuring their dimensions are uncorrelated
                if LOSS_WEIGHTS['surjectivity']:
                    _z = zprev[mask]
                    dots = _z.reshape(-1, 1) * _z.reshape(1, -1)
                    losses['surjectivity'] = torch.mean(torch.triu(dots, diagonal=1)) ** 2
                    
                    # assert self.state_dim > 1
                    # _z = zprev[mask]
                    # _z = _z / torch.norm(_z, dim=-1).unsqueeze(-1)
                    # # _loss = 1 / (1e-7 + torch.linalg.det(_z.T @ _z))
                    # # _loss = torch.norm(torch.triu(_z @ _z.T, diagonal=1))
                    # _loss = torch.abs(torch.mean(torch.triu(_z.T @ _z, diagonal=1)))
                    # losses['surjectivity'] = _loss # 2 * torch.triu(torch.abs(gram), diagonal=1).sum() / (_n ** 2 - _n) #(obs_sq_dists[_mask] / (emb_sq_dists[_mask] + 1e-3 * threshold)).mean() if _mask.sum() > 0 else torch.tensor(0.)
                    
                if LOSS_WEIGHTS['isometry']:  # check the overleaf for this one boss :)
                    diffs = zgt - zprev
                    bus = (B @ u[:-1].unsqueeze(-1)).squeeze(-1)
                    LHS = (diffs * bus).sum(dim=-1)
                    
                    axs = ((A - torch.eye(A.shape[0])) @ zprev.unsqueeze(-1)).squeeze(-1)
                    RHS = torch.norm(bus, dim=-1) ** 2 + (axs * bus).sum(dim=-1)
                    losses['isometry'] = torch.nn.functional.mse_loss(LHS, RHS)
                    
                    # _f = torch.sqrt(f)  # unsquared norms
                    # a, b = opnorm(A - torch.eye(A.shape[0])), opnorm(B)
                    # prev_norms, next_norms = _f[:-1], _f[1:]
                    # u_norms = torch.norm(u[:-1], dim=-1)
                    # with torch.no_grad():
                    #     LHS = torch.abs(next_norms - prev_norms)
                    #     RHS = a * prev_norms + b * u_norms
                    #     _mask = mask & (LHS < RHS)
                    #     LHS, RHS = LHS[_mask], RHS[_mask]
                    # d = torch.norm(zgt - zprev, dim=-1)[_mask]
                    # dists = torch.square(d - LHS)[d < LHS].sum()  # outside interval and left boundary is closest
                    # dists = dists + torch.square(d - RHS)[d > RHS].sum()  # outside interval and right boundary is closest
                    # losses['isometry'] = dists / (1 + _mask.sum())
                    
                # centeredness
                if LOSS_WEIGHTS['residual centeredness']: losses['residual centeredness'] = torch.norm((zhat - zgt).mean(dim=0)) ** 2  # sq norm of mean residual
                if LOSS_WEIGHTS['centeredness']: losses['centeredness'] = torch.norm(z.mean(dim=0)) ** 2 / (torch.norm(z, dim=-1) ** 2).mean()  # sq norm of mean embedding
                
                loss = 0.
                for k, v in losses.items(): 
                    l = LOSS_WEIGHTS[k] * v
                    loss += l
                    overall_losses[k].append(l.item())
                loss.backward(retain_graph=True)
                self.lifter_opt.step()
                
                if i_iter % print_every == 0 or i_iter == self.num_iters - 1:
                    logging.info('mean loss for iters {} - {}:'.format(i_iter - print_every, i_iter))
                    ks = sorted(overall_losses.keys())
                    for k in ks: logging.info('\t\t{}: \t{}'.format(k, np.mean(overall_losses[k][-print_every:])))

            # identify the system dynamics at the end
            with torch.no_grad():
                _x = self.normalize_x(torch.tensor(np.array(states)))
                _f = self.normalize_f(torch.tensor(np.array(costs)))
                z = jnp.array(self.lifter.encode(_x, sq_norms=_f)[0].detach().data.numpy()) * self.sqrt_fmean
                self.AB['regression'] = least_squares(z, controls, max_opnorm=MAX_OPNORM)
                self.AB['moments'] = method_of_moments(z, controls)
                if hasattr(self.lifter, 'A'):
                    _A, _B = torch.diag(self.lifter.A).detach().data.numpy(), self.lifter.B.detach().data.numpy() * self.sqrt_fmean
                    # _A, _B = self.lifter.A.detach().data.numpy(), self.lifter.B.detach().data.numpy() * self.sqrt_fmean
                    if self.lifter.latent_dim != self.lifter.state_dim:
                        _P = self.lifter.proj.detach().data.numpy()
                        _A, _B = _P @ _A @ np.linalg.pinv(_P), _P @ _B
                    self.AB['learned'] = _A, _B
                    logging.info('(LIFTER): fmean = {}'.format(fmean))
                    ret = 'regression'
                else:
                    ret = 'regression'
            
        if wordy:
            for k, v in self.AB.items():
                print('{}{} :'.format(k, ' (ret)' if k == ret else ''))
                print(summarize_lds(*v))
                print()
        
        self.trained = True
        self.A, self.B = map(lambda arr: jnp.array(arr), self.AB[ret])
        return self.A, self.B
    
    def get_state(self, 
                  obs: jnp.ndarray,
                  cost: float):
        assert obs.shape == (self.obs_dim,), (obs.shape, self.obs_dim)
        if not self.trained:
            logging.warning('({}): tried to use lifter during sysid phase. ENDING SYSID PHASE at step {}'.format(get_classname(self), self.t))
            self.end_exploration()
        if hasattr(cost, 'item'): cost = cost.item()
        
        if self.method == 'identity':
            state = obs
        elif self.method in KOOPMAN_METHODS:
            obs = np.array(obs)  # pykoopman works in numpy, not jax
            state = self.model.observables.transform(obs.reshape(1, self.obs_dim)).squeeze(0)
            state = jnp.array(state)
        elif self.method == 'nn':  # we do dl in torch, not jax
            with torch.no_grad():
                x = torch.tensor(np.array(obs), dtype=torch.float32)
                x = self.normalize_x(x)
                sq_norm = self.normalize_f(cost)
                z = self.lifter.encode(x, sq_norms=sq_norm)[0]
                z = z * self.sqrt_fmean
                state = jnp.array(z.data.numpy())
        return state
    
# ---------------------------------------------------------------------------------------------------------------------
#         SYSID WRAPPER TO EXPLORE AND FIT A SYSTEMMODEL (either first or online & jointly with controller)
# ---------------------------------------------------------------------------------------------------------------------

class LiftedController(Controller):
    def __init__(self,
                 controller: Controller,
                 lifter: SystemModel):
        # check things make sense
        assert controller.control_dim == lifter.control_dim, 'inconsistent control dims!'
        assert controller.state_dim == lifter.state_dim, 'inconsistent state dims!'
        
        super().__init__()
        self.controller = controller
        self.lifter = lifter
        self.control_dim = controller.control_dim
        self.state_dim = lifter.obs_dim
        self.stats = Stats.concatenate((controller.stats, lifter.stats))
        self._t = 0
        controller.stats = self.stats
        # lifter.stats = self.stats
        self.system_reset_hook = self.controller.system_reset_hook
        pass
    
    def get_control(self, cost: float, obs: jnp.ndarray) -> jnp.ndarray:
        assert obs.shape == (self.state_dim,), (obs.shape, self.state_dim)
        state = self.lifter.get_state(obs, cost=cost)
        return self.controller.get_control(cost, state)
    
    def update(self, state: jnp.ndarray, cost: float, control: jnp.ndarray, next_state: jnp.ndarray, next_cost: float):
        # print(jnp.linalg.norm(state[2]) ** 2, cost, '\t\t\t', jnp.linalg.norm(next_state[2]) ** 2, next_cost)  # to confirm we are seeing aligned costs and states
        state, next_state = self.lifter.get_state(state, cost=cost), self.lifter.get_state(next_state, cost=next_cost)
        self._t += 1
        return self.controller.update(state, cost, control, next_state, next_cost)
    
    @property
    def t(self): return self._t
    
    @t.setter
    def t(self, value: int): 
        self._t = value
        self.controller.t = value
        pass
    

class OfflineSysid(Controller):
    def __init__(self,
                 make_controller: Callable[[SystemModel], Controller],
                 sysid: SystemModel,
                 T0: int):        
        self.make_controller = make_controller
        self.sysid = sysid
        self.T0 = T0
        
        self.controller: Controller = None  # the moment we call get_control() and reach self.t == T0, `self.controller` will no longer be None
        
        super().__init__()
        self.t = 0
        self.state_dim = sysid.obs_dim
        self.control_dim = self.sysid.control_dim
        self.stats = self.sysid.stats
        pass

    def system_reset_hook(self):
        """
        This is a function that gets called every time the dynamical system we are controlling gets episodically reset.
        Here, we use it to alert our system model that the current trajectory has ended
        """
        if self.t < self.T0: self.sysid.end_trajectory()
        else: self.controller.system_reset_hook()
        pass
    
    def update(self, state: jnp.ndarray, cost: float, control: jnp.ndarray, next_state: jnp.ndarray, next_cost: float):
        if self.t < self.T0: return
        return self.controller.update(state, cost, control, next_state, next_cost)
    
    def get_control(self, cost: float, obs: jnp.ndarray) -> jnp.ndarray:
        assert obs.shape == (self.state_dim,), (obs.shape, self.state_dim)
        
        self.t += 1
        if self.t < self.T0: return self.sysid.explore(cost, obs)
        elif self.t == self.T0:
            logging.info('(SYSID WRAPPER) ending exploration at timestep {}'.format(self.t))
            self.sysid.end_exploration()
            
            # make the controller
            self.controller = self.make_controller(self.sysid)
            assert isinstance(self.controller, LiftedController), 'controller produced by `make_controller()` should be a LiftedController'
            assert self.controller.lifter is self.sysid, 'the controller\'s lifter should be the provided sysid, otherwise things might not work'
            assert self.controller.control_dim == self.control_dim, (self.controller.control_dim, self.control_dim)
            self.stats = self.controller.stats
            self.controller.t = self.t
            
        return self.controller.get_control(cost, obs)
    