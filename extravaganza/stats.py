import logging
from collections import namedtuple
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Tuple
import pickle as pkl

import numpy as np
import jax.numpy as jnp
import torch

from extravaganza.utils import get_classname, get_color, rescale_ax

SPECIAL_PREFIXES = ['avg']  # if one of these prefixes the name of a stat, it has auto-defined behavior

Stat = namedtuple('Stat', ['ts',
                           'values', 
                           'obj_class', 
                           'shape', 
                           'plottable'])

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
                self._avgs.append(stat_name)
                
            self._stats[key] = Stat([], [], obj_class, shape, plottable)
        return self
        
    def update(self, key, value, t: int = None):
        # make sure things make sense
        if self.aggregated: raise Exception('cannot update aggregated statistics')
        if value is None: return
        assert key in self._stats, 'please register {} with `stats.register(\'{}\', ...)`'.format(key, key)
        stat = self._stats[key]
        if stat.obj_class is not None: assert isinstance(value, stat.obj_class), (key, value.__class__)
        if stat.shape is not None and isinstance(value, (np.ndarray, jnp.ndarray, torch.Tensor)): assert value.shape == stat.shape, value.shape
        if stat.plottable is not None: assert t is not None
        
        # update stat (and any running stats)
        if t is not None: 
            stat.ts.append(t)
        stat.values.append(value)
        if key in self._avgs: 
            avg_stat = self._stats['avg {}'.format(key)]
            if t is not None: avg_stat.ts.append(t)
            avg_stat.values.append(np.mean(stat.values))
        pass
    
    def pop(self, idx: int = -1):
        if self.aggregated: raise Exception('cannot pop aggregated statistics')
        for stat in self._stats.values():
            stat.ts.pop(idx)
            stat.values.pop(idx)
        return self
    
    @staticmethod
    def concatenate(stats: Tuple):
        assert len(stats) > 0
        if len(stats) == 1: 
            logging.error('(STATS) failed to concatenate only 1 stat')
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
            logging.error('(STATS) failed to aggregate only 1 stat')
            return stats[0] 

        aggregate_stats = Stats()
        
        # accumulate all stats
        for _stats in stats:
            if _stats is None: continue
            assert isinstance(_stats, Stats), _stats.__class__
            assert not _stats.aggregated, 'cannot aggregate already aggregated stats! (it would be too confusing for me haha)'
            for k, stat in _stats._stats.items():
                # if any([prefix in k for prefix in SPECIAL_PREFIXES]): continue  # skip avgs and such
                aggregate_stats.register(k, plottable=stat.plottable, use_special_prefixes=False)
                aggregate_stats.update(k, stat.values, t=stat.ts)  # each entry in `aggregate_stats` is the entire list from a `Stats` object
                
        for k, agg_stat in aggregate_stats._stats.items():
            vs = jnp.array(agg_stat.values)
            mean, std = vs.mean(axis=0).squeeze(), vs.std(axis=0).squeeze()
            val = (mean, std, vs)
            aggregate_stats._stats[k] = aggregate_stats._stats[k]._replace(values=val)
            
            if agg_stat.plottable:
                ts = jnp.array(agg_stat.ts)
                ts = ts[0].reshape(*mean.shape)
                # assert ts.std(axis=0).mean() < 1e-4, (ts, k)
                aggregate_stats._stats[k] = aggregate_stats._stats[k]._replace(ts=ts)
        
        aggregate_stats.aggregated = True
        return aggregate_stats
    
    
    def plot(self, ax, key, fmt: str = '', label: str = None, color: str = None):            
        # assert key in self._stats, '{} not in {}'.format(key, self._stats.keys())
        if key not in self._stats: return ax
        if color is None: color = get_color(label)

        stat = self._stats[key]
        assert stat.plottable

        if self.aggregated:
            STD_CONFIDENCE = 1.
            ts = stat.ts
            means, stds, _ = stat.values
            ax.plot(ts, means, fmt, label=label, color=color)
            ax.fill_between(ts, means - STD_CONFIDENCE * stds, means + STD_CONFIDENCE * stds, alpha=0.4, facecolor=color)
        else:
            ax.plot(stat.ts, stat.values, fmt, label=label, color=color)
        ax.set_xlabel('timestep (t)')

        return ax
    
    def get_render_fns(self, 
                       ax,  # autoscales `ax` accordingly
                       xkey: str, 
                       ykey: str, 
                       slider: Tuple[None, str] = None,   # tuple of ax, key
                       num_frames: int = None,
                       label: str = None, 
                       color: str = None):
        if slider is not None: slider_ax, sliderkey = slider
        assert xkey in self._stats and ykey in self._stats and (slider is None or sliderkey in self._stats), 'unknown key(s)'
        xstat, ystat, sliderstat = self._stats[xkey], self._stats[ykey], self._stats[sliderkey] if slider is not None else None
        assert xstat.plottable and ystat.plottable and (sliderstat is None or sliderstat.plottable), 'can only render plottable statistics (stats measured over time)'
        assert np.allclose(np.array(xstat.ts), np.array(ystat.ts))
        
        # set up elements
        if color is None: color = get_color(label)
        line, = ax.plot([], [], '', color=color, lw=1.5, label=label, alpha=0.7)
        point, = ax.plot([], [], 'o', color=color)
        if sliderstat is not None: 
            sl, = slider_ax.plot([], [], '', color=color, lw=1.5, label=label)
            sp, = slider_ax.plot([], [], 'o', color=color)
        
        # get data
        ts = xstat.ts
        xs = xstat.values if not self.aggregated else xstat.values[0]
        ys = ystat.values if not self.aggregated else ystat.values[0]
        if sliderstat is not None: ss = sliderstat.values if not self.aggregated else sliderstat.values[0]
        else: ss = None
        n = len(xs)

        # get the right frames
        idxs = np.arange(n)
        if num_frames is not None:
            if num_frames < n // 2:
                rep = (n // num_frames) - 1
                idxs = [idxs[i] for i in range(0, len(idxs), rep)]
            assert len(idxs) > num_frames
            idxs = np.sort(np.random.choice(idxs, size=(num_frames,), replace=False))
        ts, xs, ys, ss = map(lambda arr: [arr[i] for i in idxs] if arr is not None else None, [ts, xs, ys, ss])

        def init_render():
            line.set_data([], [])
            point.set_data([], [])
            if sliderstat is None:
                return line, point
            else: 
                sl.set_data([], [])
                sp.set_data([], [])
                return line, point, sl, sp
        
        def render(i):  # animation function.  This is called sequentially starting from i == 0
            rescale_ax(ax, xs[i], ys[i], ignore_current_lims=i == 0)
            lidx = max(i - (len(xs) // 10), 0)  # decay existing stuff
            line.set_data(xs[lidx:i], ys[lidx:i])
            point.set_data(xs[i], ys[i])
            if sliderstat is None:
                return line, point
            else:
                rescale_ax(slider_ax, ts[i], ss[i])
                sl.set_data(ts[:i], ss[:i])
                sp.set_data(ts[i], ss[i])
                return line, point, sl, sp
            
        logging.info('(STATS): prepared to render {} over time{}'.format((xkey, ykey), ' with slider {}!'.format(sliderkey) if slider is not None else '!'))
        return init_render, render
    
    
    # everything below here is to ensure we can use the dict interface as well, affecting only values of each of our stats
    def __getitem__(self, key):
        if isinstance(key, slice):  # for slice interface!!
            assert any([stat.plottable for stat in self._stats.values()]), 'cant call time slice on stats with no time'
            max_t = np.max([np.max(stat.ts) for stat in self._stats.values() if len(stat.ts) > 0])
            lo, hi, _ = key.indices(max_t)  # ignore the step
            assert hi >= lo
            ret = deepcopy(self)
            for k, stat in ret._stats.items():
                if not stat.plottable or len(stat.ts) == 0: 
                    logging.error('({}) {} was not plottable'.format(get_classname(self), k))
                    continue
                assert len(stat.ts) == len(stat.values)
                t = jnp.array(stat.ts)
                assert jnp.allclose(t, jnp.sort(t))  # make sure its sorted
                ilo = jnp.argmax(t >= lo) if t[-1] >= lo else None
                ihi = jnp.argmax(t >= hi) if t[0] <= hi else None
                if ilo is None or ihi is None: 
                    ts = []
                    vs = []
                else:
                    if t[-1] < hi: ihi = len(stat.ts)
                    ts = stat.ts[ilo: ihi]
                    vs = stat.values[ilo: ihi]
                ret._stats[k] = ret._stats[k]._replace(ts=ts, values=vs)
            return ret
            
        elif isinstance(key, str):
            return self._stats[key].values
        elif isinstance(key, tuple):
            raise NotImplementedError('Tuple as index')
        else:
            raise TypeError('Invalid argument type: {}'.format(type(key)))
        
    def __setitem__(self, key, value):
        self.register(key)  # no-op if key in self._stats
        self._stats[key] = self._stats[key]._replace(values=value)
        pass

    def __delitem__(self, key):
        del self._stats[key]

    def __iter__(self):
        return iter({key: stat.values for key, stat in self._stats.items()})
    
    def __len__(self):
        return len(self._stats)
    
    # save and load
    def save(self, filename):
        logging.info('({}): Dumping stats to {}!'.format(get_classname(self), filename))
        with open(filename, 'r') as f:
            pkl.dump(self, f)
        pass
    
    @staticmethod
    def load(filename):
        logging.info('(STATS): Loading stats from {}!'.format(filename))
        with open(filename, 'r') as f:
            return pkl.load(f)
    