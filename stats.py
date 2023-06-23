from collections import namedtuple
from collections.abc import MutableMapping

import numpy as np
import jax.numpy as jnp
import torch

class Stats(MutableMapping):  # maps keys to the corresponding stats!
    """
    to represent statistics of an ongoing system
    """
    def __init__(self) -> None:
        self._stats = {}
        pass

    def register(self,
                 key,
                 obj_class = None,
                 shape = None,
                 plottable: bool = None):
        if key not in self._stats: self._stats[key] = namedtuple('Stat', ['value', 
                                                                          'obj_class', 
                                                                          'shape',
                                                                          'plottable']
                                                                 )([], obj_class, shape, plottable)
        return self
        
    def update(self, key, value, t=None):
        if value is None: return
        assert key in self._stats, 'please register {} with `stats.register(\'{}\', *__otherdetails__)`'.format(key, key)
        stat = self._stats[key]
        if stat.obj_class is not None: assert isinstance(value, stat.obj_class), value.__class__
        if stat.shape is not None and isinstance(value, (np.ndarray, jnp.ndarray, torch.Tensor)): assert value.shape == stat.shape, value.shape
        if stat.plottable is not None: assert t is not None
        
        if t is not None: value = (t, value)
        self._stats[key].value.append(value)
        pass

    
    def plot(self, ax, key, fmt='', label=None):
        # assert key in self._stats
        if key not in self._stats: return ax
        
        stat = self._stats[key]
        assert stat.plottable
        
        ts = [v[0] for v in stat.value]
        vals = [v[1] for v in stat.value]
        
        ax.plot(ts, vals, fmt, label=label)
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
    