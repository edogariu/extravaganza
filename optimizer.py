from functools import reduce

import torch
from torch.optim.optimizer import Optimizer

from controller import FloatController


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    
    implementation of SGD with steppable learning rate, mostly taken from https://github.com/gbaydin/hypergradient-descent/blob/master/hypergrad/sgd_hd.py 

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        hypergrad_lr (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, step_every, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SGDHD doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._params_numel = reduce(lambda total, p: total + p.numel(), self._params, 0)
        
        self.t = 0
        self.step_every = step_every
        self.closure = None

    def _gather_flat_grad_with_weight_decay(self, weight_decay=0):
        views = []
        for p in self._params:
            if p.grad is None:
                view = torch.zeros_like(p.data)
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            if weight_decay != 0:
                view.add_(weight_decay, p.data.view(-1))
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(update[offset:offset + numel].view_as(p.data), alpha=step_size)
            offset += numel
        assert offset == self._params_numel
        
    def add_closure_func(self, closure):
        self.closure = closure

    def step(self):
        """
        Performs a single optimization step.
        """
        assert self.closure is not None, '`self.closure()` needs to reevaluate the model and return the error'
        assert len(self.param_groups) == 1

        error = self.closure()

        group = self.param_groups[0]
        lr = group['lr'].item() if isinstance(group['lr'], FloatController) else group['lr']
        weight_decay = group['weight_decay']
        momentum = group['momentum'].item() if isinstance(group['momentum'], FloatController) else group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        grad = self._gather_flat_grad_with_weight_decay(weight_decay)

        # NOTE: SGDHD has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        # State initialization
        if len(state) == 0:
            state['grad_prev'] = torch.zeros_like(grad)
            state['prev_lr'] = 0
            state['prev_momentum'] = 0

        grad_prev = state['grad_prev']
        
        # LR UPDATE STEP
        if isinstance(group['lr'], FloatController) and self.t % self.step_every == 0:
            import numpy as np
            if np.isnan(lr): exit(0)
            
            grad_lr = -torch.dot(grad, grad_prev).detach().cpu().data.numpy()
            B = -torch.dot(grad_prev, grad_prev).detach().cpu().data.numpy()
            if momentum != 0 and 'momentum_buffer' in state: 
                B -= state['prev_momentum'] * torch.dot(state['momentum_buffer'], grad_prev).detach().cpu().data.numpy()
            group['lr'].step(obj=error, grad_u=grad_lr, B=B)  # with given gradients
            # group['lr'].step(obj=error, B=B)  # with estimating gradients
            # group['lr'].step(obj=error)  # with estimating gradients and system info!
        
        # MOMENTUM UPDATE STEP
        if isinstance(group['momentum'], FloatController) and self.t % self.step_every == 0 and momentum != 0 and 'momentum_buffer' in state:
            B = -state['prev_lr'] * torch.dot(state['momentum_buffer'], grad_prev).detach().cpu().data.numpy()
            group['momentum'].step(obj=error, B=B)  # with estimating gradients

        if momentum != 0:
            if 'momentum_buffer' not in state:
                buf = state['momentum_buffer'] = torch.zeros_like(grad)
                buf.mul_(momentum).add_(grad)
            else:
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(1 - dampening, grad)
            if nesterov:
                grad.add_(momentum, buf)
            else:
                grad = buf

        state['grad_prev'] = grad
        state['prev_lr'] = lr
        state['prev_momentum'] = momentum

        self._add_grad(-lr, grad)
        self.t += 1

        return error