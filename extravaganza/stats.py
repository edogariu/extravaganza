from collections import namedtuple
from collections.abc import MutableMapping
from typing import Tuple
import pickle as pkl

import numpy as np
import jax.numpy as jnp
import torch

SPECIAL_PREFIXES = ['avg']

class Stats(MutableMapping):  # maps keys to the corresponding stats!
    """
    to represent statistics of an ongoing system. 
    """
    def __init__(self) -> None:
        self._stats = {}
        self._avgs = []  # list of stats we are computing running averages of
        
        self.aggregated = False  # whether this has been aggregated
        pass

    def register(self,
                 key,
                 obj_class = None,
                 shape = None,
                 plottable: bool = None,
                 use_special_prefixes: bool = True):
        if key not in self._stats:
            if use_special_prefixes and 'avg' in key:  # for automatically calculating the cumulative mean of a stat, just register '`avg __stat_name__'`
                assert 'avg' in SPECIAL_PREFIXES
                prefix, stat_name = key.split(' ')
                assert prefix == 'avg'
                assert stat_name in self._stats, 'cannot register running average of unregistered stat {}'.format(stat_name)
                print('registering running average of {}'.format(stat_name))
                self._avgs.append(stat_name)
                
            self._stats[key] = namedtuple('Stat', ['value', 
                                                   'obj_class', 
                                                   'shape',
                                                   'plottable']
                                          )([], obj_class, shape, plottable)
        return self
        
    def update(self, key, value, t: int = None):
        if self.aggregated: raise Exception('cannot update aggregated statistics')
        if value is None: return
        assert key in self._stats, 'please register {} with `stats.register(\'{}\', *__otherdetails__)`'.format(key, key)
        stat = self._stats[key]
        if stat.obj_class is not None: assert isinstance(value, stat.obj_class), value.__class__
        if stat.shape is not None and isinstance(value, (np.ndarray, jnp.ndarray, torch.Tensor)): assert value.shape == stat.shape, value.shape
        if stat.plottable is not None: assert t is not None
        
        if t is not None: value = (t, value)
        self._stats[key].value.append(value)
        if key in self._avgs: 
            if t is not None: 
                avg = (t, np.mean([v[1] for v in self._stats[key].value]))
            else:
                avg = np.mean(self._stats[key].value)
            self._stats['avg {}'.format(key)].value.append(avg)
        pass
    
    def pop(self, idx: int = -1):
        if self.aggregated: raise Exception('cannot pop aggregated statistics')
        for stat in self._stats.values():
            stat.value.pop(idx)
        return self
    
    @staticmethod
    def concatenate(stats: Tuple):
        assert len(stats) > 0
        if len(stats) == 1: 
            print('WARNING: failed to concatenate only 1 stat')
            return stats[0]
        
        concatenated_stats = Stats()
        for s in stats:
            assert not s.aggregated, 'cannot concatenate aggregated stats'
            concatenated_stats._stats.update(s._stats)
            concatenated_stats._avgs.extend(s._avgs)
        return concatenated_stats
    
    @staticmethod
    def aggregate(stats: Tuple):
        assert len(stats) > 0
        if len(stats) == 1: 
            print('WARNING: failed to aggregate only 1 stat')
            return stats[0] 

        aggregate_stats = Stats()
        
        # accumulate all stats
        for _stats in stats:
            if _stats is None: continue
            assert isinstance(_stats, Stats), _stats.__class__
            assert not _stats.aggregated, 'cannot aggregate already aggregated stats! (it would be too confusing for me)'
            for k, stat in _stats._stats.items():
                # if any([prefix in k for prefix in SPECIAL_PREFIXES]): continue  # skip avgs and such
                if len(stat.value) == 0: continue
                if stat.plottable:
                    ts = [v[0] for v in stat.value]
                    vs = jnp.array([v[1] for v in stat.value])
                    v = (ts, vs)
                else:
                    v = jnp.array(stat.value)
                    
                aggregate_stats.register(k, plottable=stat.plottable, use_special_prefixes=False)
                aggregate_stats.update(k, v, t=0)  # dummy t dimension
                
        for k, stat in aggregate_stats._stats.items():
            if stat.plottable:
                value = [v[1] for v in stat.value if isinstance(v[1][1], jnp.ndarray)]  # ignore dummy t dimension
                ts = jnp.array([v[0] for v in value])
                vs = jnp.array([v[1] for v in value]).reshape(len(value), ts.shape[1], *((1,) if stat.shape is None else stat.shape))
                mean, std = vs.mean(axis=0), vs.std(axis=0)
                # assert ts.std(axis=0).mean() < 1e-4, (ts, k)
                ts = ts[0].reshape(*mean.shape)
                val = (ts.squeeze(), mean.squeeze(), std.squeeze(), vs)
            else:
                value = [v[1] for v in stat.value if isinstance(v[1], jnp.ndarray)]  # ignore dummy t dimension
                vs = jnp.stack(value)
                mean, std = vs.mean(axis=0), vs.std(axis=0)
                val = (mean.squeeze(), std.squeeze(), vs)
            aggregate_stats[k] = val
        
        aggregate_stats.aggregated = True
        return aggregate_stats
    
    
    def plot(self, ax, key, fmt: str = '', label: str = None):            
        # assert key in self._stats, '{} not in {}'.format(key, self._stats.keys())
        if key not in self._stats: return ax

        stat = self._stats[key]
        assert stat.plottable

        if self.aggregated:
            STD_CONFIDENCE = 1.
            ts, means, stds, _ = stat.value
            ax.plot(ts, means, fmt, label=label)
            ax.fill_between(ts, means - STD_CONFIDENCE * stds, means + STD_CONFIDENCE * stds, alpha=0.4)
        else:
            ts = [v[0] for v in stat.value]
            vals = [v[1] for v in stat.value]
            ax.plot(ts, vals, fmt, label=label)
        ax.set_xlabel('timestep (t)')

        return ax
    
    # everything below here is to ensure we can use the dict interface as well, affecting only values of each of our stats
    def __getitem__(self, key):
        return self._stats[key].value

    def __setitem__(self, key, value):
        self.register(key)  # no-op if key in self._stats
        self._stats[key] = self._stats[key]._replace(value=value)
        pass

    def __delitem__(self, key):
        del self._stats[key]

    def __iter__(self):
        return iter({key: stat.value for key, stat in self._stats.items()})
    
    def __len__(self):
        return len(self._stats)
    
    # save and load
    def save(self, filename):
        print('Dumping stats to {}!'.format(filename))
        with open(filename, 'r') as f:
            pkl.dump(self, f)
        pass
    
    @staticmethod
    def load(filename):
        print('Loading stats from {}!'.format(filename))
        with open(filename, 'r') as f:
            return pkl.load(f)
    