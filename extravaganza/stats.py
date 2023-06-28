from collections import namedtuple
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Tuple
import pickle as pkl

import numpy as np
import jax.numpy as jnp
import torch

SPECIAL_PREFIXES = ['avg']  # if one of these prefixes the name of a stat, it has auto-defined behavior

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
                 plottable: bool = False,
                 use_special_prefixes: bool = True):
        if key not in self._stats:  # if we already have it, it's a no-op
            
            if use_special_prefixes and 'avg' in key:  # for automatically calculating the cumulative mean of a stat, just register '`avg __stat_name__'`
                assert 'avg' in SPECIAL_PREFIXES
                prefix, stat_name = key.split(' ')
                assert prefix == 'avg'
                assert stat_name in self._stats, 'cannot register running average of unregistered stat {}'.format(stat_name)
                print('registering running average of {}'.format(stat_name))
                self._avgs.append(stat_name)
                
            self._stats[key] = namedtuple('Stat', ['t',
                                                   'value', 
                                                   'obj_class', 
                                                   'shape',
                                                   'plottable']
                                          )([], [], obj_class, shape, plottable)
        return self
        
    def update(self, key, value, t: int = None):
        # make sure things make sense
        if self.aggregated: raise Exception('cannot update aggregated statistics')
        if value is None: return
        assert key in self._stats, 'please register {} with `stats.register(\'{}\', ...)`'.format(key, key)
        stat = self._stats[key]
        if stat.obj_class is not None: assert isinstance(value, stat.obj_class), value.__class__
        if stat.shape is not None and isinstance(value, (np.ndarray, jnp.ndarray, torch.Tensor)): assert value.shape == stat.shape, value.shape
        if stat.plottable is not None: assert t is not None
        
        # update stat (and any running stats)
        if t is not None: stat.t.append(t)
        stat.value.append(value)
        if key in self._avgs: 
            avg_stat = self._stats['avg {}'.format(key)]
            if t is not None: avg_stat.t.append(t)
            avg_stat.value.append(np.mean(stat.value))
        pass
    
    def pop(self, idx: int = -1):
        if self.aggregated: raise Exception('cannot pop aggregated statistics')
        for stat in self._stats.values():
            stat.t.pop(idx)
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
                aggregate_stats.register(k, plottable=stat.plottable, use_special_prefixes=False)
                aggregate_stats.update(k, stat.value, t=stat.t)  # each entry in `aggregate_stats` is the entire list from a `Stats` object
                
        for k, agg_stat in aggregate_stats._stats.items():
            vs = jnp.array(agg_stat.value)
            mean, std = vs.mean(axis=0), vs.std(axis=0)
            val = (mean.squeeze(), std.squeeze(), vs)
            aggregate_stats._stats[k].value = val
            
            if agg_stat.plottable:
                ts = jnp.array(agg_stat.t)
                ts = ts[0].reshape(*mean.shape)
                # assert ts.std(axis=0).mean() < 1e-4, (ts, k)
                aggregate_stats._stats[k].t = ts
        
        aggregate_stats.aggregated = True
        return aggregate_stats
    
    
    def plot(self, ax, key, fmt: str = '', label: str = None):            
        # assert key in self._stats, '{} not in {}'.format(key, self._stats.keys())
        if key not in self._stats: return ax

        stat = self._stats[key]
        assert stat.plottable

        if self.aggregated:
            STD_CONFIDENCE = 1.
            ts = stat.t
            means, stds, _ = stat.value
            ax.plot(ts, means, fmt, label=label)
            ax.fill_between(ts, means - STD_CONFIDENCE * stds, means + STD_CONFIDENCE * stds, alpha=0.4)
        else:
            ax.plot(stat.t, stat.value, fmt, label=label)
        ax.set_xlabel('timestep (t)')

        return ax
    
    # everything below here is to ensure we can use the dict interface as well, affecting only values of each of our stats
    def __getitem__(self, key):
        if isinstance(key, slice):  # for slice interface!!
            assert any([stat.plottable for stat in self._stats.values()]), 'cant call time slice on stats with no time'
            max_t = np.max([np.max(stat.t) for stat in self._stats.values() if len(stat.t) > 0])
            lo, hi, _ = key.indices(max_t)  # ignore the step
            assert hi >= lo
            ret = deepcopy(self)
            for k, stat in ret._stats.items():
                if not stat.plottable or len(stat.t) == 0: 
                    print(k, ' was not plottable')
                    continue
                assert len(stat.t) == len(stat.value)
                t = jnp.array(stat.t)
                assert jnp.allclose(t, jnp.sort(t))  # make sure its sorted
                ilo = jnp.argmax(t >= lo) if t[-1] >= lo else None
                ihi = jnp.argmax(t >= hi) if t[0] <= hi else None
                if ilo is None or ihi is None: 
                    ts = []
                    vs = []
                else:
                    if t[-1] < hi: ihi = len(stat.t)
                    ts = stat.t[ilo: ihi]
                    vs = stat.value[ilo: ihi]
                ret._stats[k] = ret._stats[k]._replace(t=ts, value=vs)
            return ret
            
        elif isinstance(key, str):
            return self._stats[key].value
        elif isinstance(key, tuple):
            raise NotImplementedError('Tuple as index')
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))
        
    # def __getitem__(self, key):
    #     return self._stats[key].value

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
    