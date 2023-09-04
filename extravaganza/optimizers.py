from typing import Callable
import torch.optim as optim

class SGD(optim.SGD):
    def step(self, closure: Callable[[], float] | None = ...) -> float | None:
        l = super().step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None: