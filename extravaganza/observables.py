from typing import Callable

import jax
import jax.numpy as jnp

from extravaganza.utils import set_seed, jkey, timestep_embedding

"""
This class defines an "observable", which is some real-valued vector observation that is made using the current trajectory 
as input.
 
Examples:
- If the system is fully observable, we might just let the observable return the newest state. 
- if the system is partially observable, we pick a random matrix `C \in R^{d_obs, d_state}` and always return `C` times the newest state.
- If the system is "unobservable", we might use the past `hh` costs and controls as our observation vector.

Note that the `obs_func` must define what to do in the beginning when the trajectory is short (it will never be empty, but may only contain )
"""

class Trajectory:
    def __init__(self):
        self.x = []  # state we are in
        self.f = []  # cost of state we are in
        self.u = []  # control we play from that state
    
    def pad(self, min_len: int, control_dim: int = None, state_dim: int = None):
        while len(self.f) < min_len: self.f.append(0.)
        while len(self.x) < min_len: self.x.append(jnp.zeros(state_dim) if state_dim is not None else None)
        while len(self.u) < min_len: self.u.append(jnp.zeros(control_dim) if control_dim is not None else None)
    
    def add_state(self, cost: float, state: jnp.ndarray):
        if hasattr(cost, 'item'): cost = cost.item()
        # if abs(cost) > 1e2: cost /= (abs(cost) / 1e2)
        self.f.append(cost)
        self.x.append(state)
    
    def add_control(self, control: jnp.ndarray):
        self.u.append(control)
        
    def __len__(self):
        assert len(self.x) == len(self.u) and len(self.u) == len(self.f), (len(self.x), len(self.u), len(self.f))
        return len(self.x)

class Observable:
    
    def __init__(self, obs_func, obs_dim) -> None:
        self.obs_func = obs_func
        self.obs_dim = obs_dim
        pass
    
    obs_func: Callable[[Trajectory], jnp.ndarray]
    obs_dim: int
    
    def __call__(self, trajectory: Trajectory):
        """
        Creates an observation vector from the trajectory. Note that we are given this trajectory BEFORE deciding what control to play. 
        This is because the observation we make here will be what is used to pick the control.

        Parameters
        ----------
        trajectory : Trajectory
        """
        obs = self.obs_func(trajectory)
        return obs
    
    
class TimeDelayedObservation(Observable):
    def __init__(self, 
                 hh: int,
                 use_states: bool,
                 use_controls: bool,
                 use_costs: bool,
                 use_cost_diffs: bool,
                 use_time: bool,
                 control_dim: int = None,
                 state_dim: int = None,
                 time_embedding_dim: int = 8):
        
        assert any([use_states, use_controls, use_costs, use_time]), 'must use something!'
        
        obs_dim = 0
        if use_states: 
            assert state_dim is not None, 'must provide state dim to use states in observable'
            obs_dim += hh * state_dim
        if use_controls:
            assert control_dim is not None, 'must provide control dim to use controls in observable'
            obs_dim += hh * control_dim
        if use_costs:
            obs_dim += hh * 1
        if use_cost_diffs:
            obs_dim += hh * 1
        if use_time:
            obs_dim += time_embedding_dim
            
        def obs_func(trajectory: Trajectory):
            if any([len(l) <= hh for l in [trajectory.x, trajectory.u, trajectory.f]]): trajectory.pad(hh + 1, control_dim, state_dim)
            obs = []
            if use_states: 
                t = trajectory.x[-hh:]
                for _t in t: assert _t.shape == (state_dim,), (_t.shape, state_dim)
                obs.extend(t)
            if use_controls: 
                t = trajectory.u[-hh:]
                for _t in t: assert _t.shape == (control_dim,), (_t.shape, control_dim)
                obs.extend(t)
            if use_costs: 
                t = trajectory.f[-hh:]
                for _t in t: assert isinstance(_t, float), _t.__class__
                obs.append(jnp.array(t))
            if use_cost_diffs: 
                for _t in trajectory.f[-hh:]: assert isinstance(_t, float), _t.__class__
                t = jnp.stack(trajectory.f[-hh:], axis=0) - jnp.stack(trajectory.f[-hh - 1:-1], axis=0)
                obs.append(t)
            if use_time: 
                t = (1 + len(trajectory.u)) // 3
                emb = timestep_embedding(jnp.array([[t],]), embedding_dim=time_embedding_dim, method='sin')
                assert emb.shape == (time_embedding_dim,)
                obs.append(emb)
                
            obs = jnp.concatenate(obs, axis=-1)
            assert obs.shape == (obs_dim,), (obs.shape, obs_dim)
            return obs
        
        super().__init__(obs_func, obs_dim)
        self.hh = hh
        
class FullObservation(Observable):
    def __init__(self, 
                 state_dim: int) -> None:
        
        def obs_func(trajectory: Trajectory):
            return trajectory.x[-1]  # most recent state
        
        super().__init__(obs_func, state_dim)
 
class PartialObservation(Observable):
    def __init__(self,
                 obs_dim: int,
                 state_dim: int,
                 seed: int = None) -> None:
        set_seed(seed)
        self.C = jax.random.normal(jkey(), shape=(obs_dim, state_dim))
        
        def obs_func(trajectory: Trajectory):
            return self.C @ trajectory.x[-1]  # transfromation of last state
        
        super().__init__(obs_func, obs_dim)
        
if __name__ == '__main__':
    """
    test that tings work
    """
    from copy import deepcopy
    
    from extravaganza.dynamical_systems import LDS
    from extravaganza.utils import jkey, sample
    
    do = 5
    hh = 6
    ds = 3
    du = 8
    system = LDS(ds, du, 'none', 'quad')
    
    full = FullObservation(ds)
    partial = PartialObservation(do, ds)
    use_states, use_controls, use_costs, use_time = True, True, True, True
    td = TimeDelayedObservation(hh, control_dim=du, state_dim=ds, use_states=use_states, use_costs=use_costs, use_controls=use_controls, use_time=use_time)
    custom = Observable(lambda traj: traj.x[-1].mean().reshape(1), 1)
    
    trajs = []
    traj = Trajectory()
    state = system.initial_state
    cost = system.cost_fn(state, jnp.zeros(du))
    traj.add_state(cost, state)
    for _ in range(100):  # make a trajectory of len 100
        control = sample(jkey(), (du,))
        cost, state = system.interact(control)
        traj.add_state(cost, state)
        traj.add_control(control)
        trajs.append(deepcopy(traj))
    
    # make sure observations were ok the whole way through
    if use_costs: assert jnp.allclose(td.norm_fn(td(traj)), cost ** 2)    
    def check(traj):
        assert full(traj).shape == (full.obs_dim,) and jnp.allclose(full(traj), traj.x[-1]), full(traj).shape
        assert partial(traj).shape == (partial.obs_dim,) and jnp.allclose(partial(traj), partial.C @ traj.x[-1]), partial(traj).shape
        assert td(traj).shape == (td.obs_dim,)
        assert custom(traj).shape == (custom.obs_dim,) and custom(traj) == traj.x[-1].mean()
    for traj in trajs: check(traj)

    print('yippee!')
    