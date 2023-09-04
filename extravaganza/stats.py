import logging
from collections import namedtuple, defaultdict
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Tuple, Callable
import pickle as pkl

import numpy as np
import jax.numpy as jnp
import torch

from extravaganza.utils import get_classname, get_color, rescale_ax

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
        self._attachments = defaultdict(list)  # the key is `key_to_attach_to`, and the value is a list of dicts with keys `('key', 'fn')`
        self.aggregated = False  # whether this has been aggregated
        pass

    def register(self,
                 key: str,
                 obj_class = None,
                 shape = None,
                 plottable: bool = True):
        """
        register a new stat that we will be manually updating with `self.update()`
        """
        assert key not in self._stats, 'key {} already in stats!'.format(key)
                
        self._stats[key] = Stat([], [], obj_class, shape, plottable)
        return self
    
    def attach(self,
               key: str,  # new key
               key_to_attach_to: str,  # old key
               fn: Callable[[object], object],
               
               new_obj_class = None,
               new_shape = None):
        """
        Register a stat that we will automatically update via calls to `fn `.
        The given `fn` should take as input an object of class `self._stats[key_to_attach_to].obj_class` with shape `self._stats[key_to_attach_to].shape`,
        and return an object of class `new_obj_class` of shape `new_shape`.
        If the old stat is plottable, then the new stat will be as well.
        """
        assert key not in self._stats and key not in self._attachments, '{} is already in the stats!'.format(key)
        assert key_to_attach_to in self._stats, '{} isnt in the stats yet'.format(key_to_attach_to)
        
        self.register(key, obj_class=new_obj_class, shape=new_shape, plottable=self._stats[key_to_attach_to].plottable)
        self._attachments[key_to_attach_to].append({'key': key, 'fn': fn})
        return self
    
        
    def update(self, key, value, t: int = None):
        # make sure things make sense
        if self.aggregated: raise Exception('cannot update aggregated statistics')
        if value is None: return
        assert key in self._stats, 'please register {} with `stats.register(\'{}\', ...)`'.format(key, key)
        stat = self._stats[key]
        if stat.obj_class is not None: assert isinstance(value, stat.obj_class), (key, value.__class__)
        if stat.shape is not None and isinstance(value, (np.ndarray, jnp.ndarray, torch.Tensor)): assert value.shape == stat.shape, (value.shape, stat.shape)
        if stat.plottable: assert t is not None
        
        # update stat
        stat.values.append(value)
        if stat.plottable: stat.ts.append(t)
        
        # update stat's attachments
        for attachment in self._attachments[key]: self.update(attachment['key'], value=attachment['fn'](value), t=t)
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
        
        _stats = []
        for i in range(len(stats) - 1): 
            if stats[i] not in stats[i+1:]:
                _stats.append(stats[i])
        _stats.append(stats[-1])
        
        concatenated_stats = Stats()
        for s in stats:
            assert not s.aggregated, 'cannot concatenate aggregated stats'
            if not set(concatenated_stats._stats.keys()).isdisjoint(set(s._stats.keys())): 
                print(set(concatenated_stats._stats.keys()).intersection(set(s._stats.keys())))
            concatenated_stats._stats.update(s._stats)
            concatenated_stats._attachments.update(s._attachments)
        return concatenated_stats
    
    @staticmethod
    def aggregate(stats: Tuple):
        N = len(stats)
        assert N > 0
        if N == 1: 
            logging.warning('(STATS) tried to aggregate only 1 stat')
            return stats[0] 

        aggregate_stats = Stats()
        
        # accumulate all stats
        for _stats in stats:
            if _stats is None: continue
            assert isinstance(_stats, Stats), _stats.__class__
            assert not _stats.aggregated, 'cannot aggregate already aggregated stats! (it would be too confusing for me haha)'
            for k, stat in _stats._stats.items():
                if k not in aggregate_stats._stats: aggregate_stats.register(k, obj_class=list, plottable=stat.plottable)
                aggregate_stats.update(k, stat.values, t=stat.ts)  # each entry in `aggregate_stats` is the entire list from a `Stats` object
            aggregate_stats._attachments.update(_stats._attachments)
                
        for k, agg_stat in aggregate_stats._stats.items():
            try:
                l = min([len(v) for v in agg_stat.values])
                if l == 0: continue
                for i in range(len(agg_stat.values)): agg_stat.values[i] = agg_stat.values[i][:l]
                vs = jnp.array(agg_stat.values)
                means, stds = vs.mean(axis=0).squeeze(), vs.std(axis=0).squeeze()
                val = {'means': means, 'stds': stds, 'vs': vs}
                aggregate_stats._stats[k] = aggregate_stats._stats[k]._replace(values=val)
                
                if agg_stat.plottable:
                    for i in range(len(agg_stat.ts)): agg_stat.ts[i] = agg_stat.ts[i][:l]
                    ts = jnp.array(agg_stat.ts).reshape(N, -1)
                    assert ts.std(axis=0).mean() < 1e-4, (ts, k)
                    aggregate_stats._stats[k] = aggregate_stats._stats[k]._replace(ts=ts[0])  # just use the first set of times, they should all be the same
            except Exception as e:
                logging.error('{} had an error'.format(k))
                raise e
        
        aggregate_stats.aggregated = True
        return aggregate_stats
    
    
    def plot(self, ax, key, 
             plot_idx: int = None, plot_norm: bool = False, plot_cummean: bool = False,
             fmt: str = '', label: str = None, color: str = None):            
        # make sure things make sense
        assert not (plot_idx is not None and plot_norm), 'cant plot both index {} and the norm!'.format(plot_idx)
        # assert key in self._stats, '{} not in {}'.format(key, self._stats.keys())
        if key not in self._stats: return ax
        if color is None: color = get_color(label)
        stat = self._stats[key]
        assert stat.plottable

        if self.aggregated:
            STD_CONFIDENCE = 1.
            ts = stat.ts
            means, stds, vs = stat.values['means'], stat.values['stds'], stat.values['vs']
            print(stds.shape, stds.mean())
            
            if plot_idx is not None:  # select an index to plot; it will have the respective slices of the mean and std
                assert means.ndim == 2 and stds.ndim == 2, 'we dont have multidimensional stats to plot idx {} of in the first place'.format(plot_idx)
                means, stds = means[:, plot_idx], stds[:, plot_idx]
            elif plot_norm:  # we have to do it this way since norm of avg vector is not the avg of the vector norms
                assert vs.ndim == 3, 'we dont have multidimensional stats to plot the norm of in the first place'  # (N, T, D)
                norms = jnp.linalg.norm(vs, axis=-1)
                means, stds = norms.mean(axis=0).squeeze(), norms.std(axis=0).squeeze()
                
            if plot_cummean:  # compute moments of the cumulative mean of the RV
                means = jnp.cumsum(means, axis=0) / jnp.arange(1, means.shape[0] + 1)
                stds = jnp.sqrt(jnp.cumsum(stds ** 2, axis=0)) / jnp.arange(1, stds.shape[0] + 1)
            
            ax.plot(ts, means, fmt, label=label, color=color)
            ax.fill_between(ts, means - STD_CONFIDENCE * stds, means + STD_CONFIDENCE * stds, alpha=0.4, facecolor=color)
        else:
            ts, vals = stat.ts, np.array(stat.values)
            if plot_idx is not None: vals = vals[:, plot_idx]
            elif plot_norm: vals = np.linalg.norm(vals, dim=-1)
            if plot_cummean: vals = np.cumsum(vals) / np.arange(1, len(vals) + 1)
            ax.plot(stat.ts, vals, fmt, label=label, color=color)
        
        ax.set_xlabel('timestep (t)')
        return ax
    
    # def get_render_fns(self, 
    #                    ax,  # autoscales `ax` accordingly
    #                    xkey: str, 
    #                    ykey: str, 
    #                    slider: Tuple[None, str] = None,   # tuple of ax, key
    #                    num_frames: int = None,
    #                    label: str = None, 
    #                    color: str = None):
    #     if slider is not None: slider_ax, sliderkey = slider
    #     assert xkey in self._stats and ykey in self._stats and (slider is None or sliderkey in self._stats), 'unknown key(s)'
    #     xstat, ystat, sliderstat = self._stats[xkey], self._stats[ykey], self._stats[sliderkey] if slider is not None else None
    #     assert xstat.plottable and ystat.plottable and (sliderstat is None or sliderstat.plottable), 'can only render plottable statistics (stats measured over time)'
    #     assert np.allclose(np.array(xstat.ts), np.array(ystat.ts))
        
    #     # set up elements
    #     if color is None: color = get_color(label)
    #     line, = ax.plot([], [], '', color=color, lw=1.5, label=label, alpha=0.7)
    #     point, = ax.plot([], [], 'o', color=color)
    #     if sliderstat is not None: 
    #         sl, = slider_ax.plot([], [], '', color=color, lw=1.5, label=label)
    #         sp, = slider_ax.plot([], [], 'o', color=color)
        
    #     # get data
    #     ts = xstat.ts
    #     xs = xstat.values if not self.aggregated else xstat.values[0]
    #     ys = ystat.values if not self.aggregated else ystat.values[0]
    #     if sliderstat is not None: ss = sliderstat.values if not self.aggregated else sliderstat.values[0]
    #     else: ss = None
    #     n = len(xs)

    #     # get the right frames
    #     idxs = np.arange(n)
    #     if num_frames is not None:
    #         if num_frames < n // 2:
    #             rep = (n // num_frames) - 1
    #             idxs = [idxs[i] for i in range(0, len(idxs), rep)]
    #         assert len(idxs) > num_frames
    #         idxs = np.sort(np.random.choice(idxs, size=(num_frames,), replace=False))
    #     ts, xs, ys, ss = map(lambda arr: [arr[i] for i in idxs] if arr is not None else None, [ts, xs, ys, ss])

    #     def init_render():
    #         line.set_data([], [])
    #         point.set_data([], [])
    #         if sliderstat is None:
    #             return line, point
    #         else: 
    #             sl.set_data([], [])
    #             sp.set_data([], [])
    #             return line, point, sl, sp
        
    #     def render(i):  # animation function.  This is called sequentially starting from i == 0
    #         rescale_ax(ax, xs[i], ys[i], ignore_current_lims=i == 0)
    #         lidx = max(i - (len(xs) // 10), 0)  # decay existing stuff
    #         line.set_data(xs[lidx:i], ys[lidx:i])
    #         point.set_data(xs[i], ys[i])
    #         if sliderstat is None:
    #             return line, point
    #         else:
    #             rescale_ax(slider_ax, ts[i], ss[i])
    #             sl.set_data(ts[:i], ss[:i])
    #             sp.set_data(ts[i], ss[i])
    #             return line, point, sl, sp
            
    #     logging.info('(STATS): prepared to render {} over time{}'.format((xkey, ykey), ' with slider {}!'.format(sliderkey) if slider is not None else '!'))
    #     return init_render, render
    
    
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
                if len(stat.ts) != len(stat.values):
                    logging.warning('(STATS) weird stuff going on with {}'.format(k))
                    continue
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
    