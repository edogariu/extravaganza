import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Tuple
from copy import deepcopy

import numpy as np
import torch
import pykoopman as pk

from extravaganza.models import TorchMLP, TorchLifter
from extravaganza.utils import set_seed, least_squares, method_of_moments, summarize_lds, dare_gain

LOSS_WEIGHTS = {
    # touch these
    'isometry': 0,
    'nontrivial': 1e-3,
    'jac': 0,
    'dot': 0,
    'vmf': 0,
    'dare': 0,
    'dare norms': 0,
    'jac_conditioning': 0,
    'l2 linearization': 1,
    'l2 linearization relative': 0,
    'l1 linearization': 0,
    'reconstruction': 0,
    'injectivity': 0,
    'surjectivity': 0,
    'cpc': 0,
    'guess the control': 1,
    'simplification': 0,
    'residual centeredness': 0,
    'centeredness': 0,
    
    # dont touch these
    'consistency': 0,
    'proj_isometry': 1,
    'proj_error': 1,
}

def _prep_dataset(dataset, hh: int, normalize: bool):
    # prep dataset
    os, us, fs = np.array(dataset.os), np.array(dataset.us), np.array(dataset.fs)
    o_mean, o_std, f_mean = np.mean(os, axis=0), np.std(os, axis=0), np.mean(fs)
    
    if normalize:
        os = (os - o_mean) / o_std
        fs = fs / f_mean
        normalize_fn = lambda o, f: ((o - o_mean) / o_std, f / f_mean)
    else:
        normalize_fn = lambda o, f: (o, f)
    
    # we assume that the indices in `reset_idxs` are the indices of the start of a new trajectory
    # we create a boolean mask of given length that is True everywhere except within `hh` of a reset
    idxs = np.array(dataset.reset_idxs) - 1
    mask = np.ones(os.shape[0], dtype=bool)
    for idx in idxs: mask[max(0, idx - hh + 1): idx + hh + 1] = False  # anything within +/- hh of a reset must go
    mask[:hh] = False  # the first hh must go
    
    return os, us, fs, mask, normalize_fn
    

class Lifter:
    obs_dim: int
    state_dim: int
    control_dim: int
    AB: Dict[str, Tuple[np.ndarray, np.ndarray]]
    
    @abstractmethod
    def get_state(self, 
                  obs: np.ndarray,
                  cost: float) -> np.ndarray:
        pass
    
    
class Identity(Lifter):
    def __init__(self,
                 obs_dim: int,
                 control_dim: int,
                 normalize: bool):
        self.obs_dim = self.state_dim = obs_dim
        self.control_dim = control_dim
        self.normalize = normalize
        self.AB = {}
        
    def train(self, dataset):
        os, us, fs, mask, self.normalize_fn = _prep_dataset(dataset, 1, self.normalize)
        assert os.shape[-1] == self.obs_dim and us.shape[-1] == self.control_dim
        
        self.AB['regression'] = least_squares(os, us, mask=mask[:-1]) 
        self.AB['moments'] = method_of_moments(os, us, mask=mask[:-1])           
        pass
    
    def get_state(self, obs: np.ndarray, cost: float) -> np.ndarray:
        assert obs.shape == (self.obs_dim,)
        obs, cost = self.normalize_fn(obs, cost)
        return obs
    
    
class Koopman(Lifter):
    def __init__(self,
                 obs_dim: int,
                 control_dim: int,
                 state_dim: int,
                 method: str,
                 normalize: bool):
        self.obs_dim = obs_dim
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.normalize = normalize
        
        assert method in ['fourier', 'polynomial', 'rbf']
        if method == 'fourier': assert (self.state_dim - self.obs_dim) % 2 == 0, '`state_dim - obs_dim` must be even (not {}) for Koopman fourier methods!'.format(self.state_dim - self.obs_dim)
        if method == 'polynomial': raise NotImplementedError('oopsie, for some reason polynomial observations isnt working yet')

        observables = {
            'polynomial': pk.observables.Polynomial(degree=3),
            'rbf': pk.observables.RadialBasisFunction(rbf_type='gauss', n_centers=self.state_dim - self.obs_dim, include_state=True),
            'fourier': pk.observables.RandomFourierFeatures(D=(self.state_dim - self.obs_dim) // 2, include_state=True),
        }
        self.model = pk.Koopman(observables=observables[method], regressor=pk.regression.EDMDc())
        self.AB = {}
        
    def train(self, dataset):
        os, us, fs, mask, self.normalize_fn = _prep_dataset(dataset, 1, self.normalize)
        assert os.shape[-1] == self.obs_dim and us.shape[-1] == self.control_dim
        self.model.fit(x=os, u=us)
        z = self.model.observables.transform(os)
        
        self.AB['regression'] = least_squares(z, us, mask=mask[:-1]) 
        self.AB['moments'] = method_of_moments(z, us, mask=mask[:-1])           
        self.AB['koopman'] = (self.model.A.reshape(self.state_dim, self.state_dim), self.model.B.reshape(self.state_dim, self.control_dim))
        pass
        
    def get_state(self, obs: np.ndarray, cost: float) -> np.ndarray:
        assert obs.shape == (self.obs_dim,)
        obs, cost = self.normalize_fn(obs, cost)
        state = self.model.observables.transform(obs.reshape(1, self.obs_dim)).squeeze(0)
        return state
    
        
class NN(Lifter):
    def __init__(self,
                 obs_dim: int,
                 control_dim: int,
                 state_dim: int,
                 normalize: bool,
                 
                 # nn hyperparams
                 latent_dim: int,
                 AB_method: str,
                 depth: int,  # depth of NN
                 sigma: float = 0.,
                 deterministic: bool = True,
                 isometric: bool = False,
                 lifter_lr: float = 0.001,
                 hh: int = 1,  # how many data points after a reset to ignore
                 
                 seed: int = None):

        set_seed(seed)
        
        self.obs_dim = obs_dim
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.normalize = normalize
        self.hh = hh
        
        for i in range(self.hh): 
            for k in ['l2 linearization', 'l2 linearization relative', 'l1 linearization', 'simplification']:
                LOSS_WEIGHTS[k + ' ' + str(i)] = LOSS_WEIGHTS[k]
        
        if latent_dim is None: latent_dim = self.state_dim
        assert latent_dim >= self.state_dim  # always be linearizing in a higher dimension
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
        self.lifter_opt = torch.optim.AdamW(self.lifter.parameters(), lr=lifter_lr, weight_decay=1e-5)
        self.AB = {}
        pass
    
    def train_step(self, xs, us, fs, mask=None):
        self.lifter_opt.zero_grad()
                
        (z, zhat), (A, B), losses = self.lifter.get_embs_and_losses(xs, us, sq_norms=fs, mask=mask)   
        zprev, zgt = z[:-1], z[1:]
        
        # how well the squared norm represents cost
        simp, l2, l1 = LOSS_WEIGHTS['simplification'], LOSS_WEIGHTS['l2 linearization'] or LOSS_WEIGHTS['l2 linearization relative'], LOSS_WEIGHTS['l1 linearization']
        if any((simp, l2, l2)): 
            if simp: losses['simplification'] = torch.nn.functional.mse_loss(torch.norm(zprev[mask], dim=-1) ** 2, fs[:-1][mask])
            _z = z[:-self.hh]
            for i in range(self.hh):
                l, r = i, len(z) - self.hh + i
                _m = mask[l:r]
                _z = (A @ _z.unsqueeze(-1) + B @ us[l:r].unsqueeze(-1)).squeeze(-1)
                if l2: 
                    if LOSS_WEIGHTS['l2 linearization']: losses[f'l2 linearization {i}'] = torch.nn.functional.mse_loss(_z[_m], z[l+1:r+1][_m])
                    if LOSS_WEIGHTS['l2 linearization relative']: losses[f'l2 linearization relative {i}'] = torch.mean((torch.norm(_z[_m] - z[l+1:r+1][_m], dim=-1) ** 2) / torch.norm(z[l+1:r+1][_m], dim=-1))
                if l1: losses[f'l1 linearization {i}'] = torch.nn.functional.l1_loss(_z[_m], z[l+1:r+1][_m])
                if simp: losses[f'simplification {i}'] = torch.nn.functional.mse_loss(torch.norm(_z[_m], dim=-1) ** 2, fs[l+1:r+1][_m])
        
        # if LOSS_WEIGHTS['dot']:  # compares empirical expectation to the theoretical one
        #     diffs = (zgt - zprev)[mask]
        #     diffs = (B.T @ diffs.unsqueeze(-1)).squeeze(-1)
        #     RHS = (self.explorer.exploration_scales.item() * torch.norm(B, p='fro')) ** 2                    
        #     LHS = (diffs * us[:-1][mask]).sum(dim=-1).mean()
        #     losses['dot'] = torch.nn.functional.mse_loss(LHS, RHS)
        
        # how well we can reproduce the controls we used --   znext = A @ z + B @ u  ->  B^-1_left @ (znext - A @ z) = u
        if LOSS_WEIGHTS['guess the control']:
            pinv = torch.linalg.pinv(B)
            assert torch.allclose(pinv @ B, torch.eye(B.shape[1]), rtol=1e-1, atol=1e-1), (pinv @ B)
            # pinv = torch.clamp(pinv, -1e2, 1e2)
            uhat = (pinv @ (zgt.unsqueeze(-1) - (A @ zprev.unsqueeze(-1)))).squeeze(-1) 
            losses['guess the control'] = torch.nn.functional.mse_loss(uhat[mask], us[:-1][mask])
        
        # injectivity proxy to penalize instances where two observations are distinct but are embedded similarly
        if LOSS_WEIGHTS['injectivity']:
            _threshold = 1e-2
            _x, _z = xs[:-1][mask], zprev[mask]
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
            
        if LOSS_WEIGHTS['dare']:  # we want dare controls to be close to what we played iff cost decreased0
            try:
                K = dare_gain(A, B)
                dare_controls = (-K @ zprev.unsqueeze(-1)).squeeze(-1)
                _d, _u = dare_controls, us[:-1]
                dots = (_d * _u).sum(dim=-1)
                cost_diffs = fs[1:] - fs[:-1]
                loss = (cost_diffs[mask] * dots[mask]).mean()  # where cost_diffs is very negative, we want controls to be aligned. where cost_diffs is very positive, we want controls to be antialigned. if costs didnt change, no info
                losses['dare'] = loss
                losses['dare norms'] = torch.norm(dare_controls[mask].mean(dim=0)) ** 2
            except:
                pass
            
        if LOSS_WEIGHTS['isometry']:  # check the overleaf for this one boss :)
            diffs = zgt - zprev
            bus = (B @ us[:-1].unsqueeze(-1)).squeeze(-1)
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
        if LOSS_WEIGHTS['centeredness']: losses['centeredness'] = torch.norm(z.mean(dim=0)) ** 2  # sq norm of mean embedding
        
        loss = 0.
        for k, v in losses.items(): 
            l = LOSS_WEIGHTS[k] * v
            loss += l
            losses[k] = l.item()
        loss.backward(retain_graph=True)
        self.lifter_opt.step()
        return losses
    
    def train(self, dataset, num_iters: int, batch_size: int):
        # prep dataset
        os, us, fs, mask, self.normalize_fn = _prep_dataset(dataset, self.hh, self.normalize)
        assert os.shape[-1] == self.obs_dim and us.shape[-1] == self.control_dim
        
        # convert to pytorch tensors
        X, U, F = map(lambda arr: torch.tensor(arr).float(), [os, us, fs])  # convert to tensors
        MASK = torch.tensor(mask)
        if batch_size > 0: assert batch_size < X.shape[0]
        assert X.shape[1] == self.lifter.obs_dim, (X.shape[1], self.lifter.obs_dim)
        
        # train the model
        print_every = max(num_iters // 10, 1)
        logging.info('training!')
        overall_losses = defaultdict(list)
        for i_iter in range(num_iters):
            # sample a 'batch', i.e. a slice of the training trajectory
            idx = np.random.randint(X.shape[0] - batch_size - 1)
            sl = slice(idx, idx + batch_size, 1)
            losses = self.train_step(X[sl], U[sl], F[sl], mask=MASK[sl][:-1])
            for k, v in losses.items():
                overall_losses[k].append(v)
        
            if i_iter % print_every == 0 or i_iter == num_iters - 1:
                logging.info('mean loss for iters {} - {}:'.format(i_iter - print_every, i_iter))
                ks = sorted(overall_losses.keys())
                for k in ks: logging.info('\t\t{}: \t{}'.format(k, np.mean(overall_losses[k][-print_every:])))

        # identify the system dynamics at the end
        with torch.no_grad():
            z = self.lifter.encode(X, sq_norms=F)[0].detach().data.numpy()
            self.AB['regression'] = least_squares(z, us, mask=mask[:-1])
            self.AB['moments'] = method_of_moments(z, us, mask=mask[:-1])
            if hasattr(self.lifter, 'A'):
                # _A, _B = torch.diag(self.lifter.A).detach().data.numpy(), self.lifter.B.detach().data.numpy()
                _A, _B = self.lifter.A.detach().data.numpy(), self.lifter.B.detach().data.numpy()
                if self.lifter.latent_dim != self.lifter.state_dim:
                    _P = self.lifter.proj.detach().data.numpy()
                    _A, _B = _P @ _A @ np.linalg.pinv(_P), _P @ _B
                self.AB['learned'] = _A, _B
        
        for k, v in self.AB.items():
            print('{}:'.format(k))
            print(summarize_lds(*v))
            print()
        pass

    def get_state(self, 
                  obs: np.ndarray,
                  cost: float):
        assert obs.shape == (self.obs_dim,), (obs.shape, self.obs_dim)
        obs, cost = self.normalize_fn(obs, cost)
        if hasattr(cost, 'item'): cost = cost.item()
        
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32)
            z = self.lifter.encode(x, sq_norms=cost)[0]
            state = z.data.numpy()
        return state
