import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def inv_sigmoid(s):
    return np.log(s / (1 - s + 1e-8))

def d_sigmoid(t):
    s = sigmoid(t)
    return s * (1 - s)

def rescale(t, bounds, do_rescale, use_sigmoid):
    """
    rescales from `[0, 1] -> [tmin, tmax]`
    """
    if not do_rescale: return t
    tmin, tmax = bounds
    if use_sigmoid: t = sigmoid(t)
    return tmin + (tmax - tmin) * t

def d_rescale(t, bounds, do_rescale, use_sigmoid):
    if not do_rescale: return 1
    tmin, tmax = bounds
    d = tmax - tmin
    if use_sigmoid: d *= d_sigmoid(t)
    return d

def inv_rescale(s, bounds, do_rescale, use_sigmoid):
    if not do_rescale: return s
    tmin, tmax = bounds
    t = (s - tmin) / (tmax - tmin)
    if use_sigmoid: t = inv_sigmoid(t)
    return t

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
