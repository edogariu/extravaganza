import logging
import numpy as np

from extravaganza.controllers import Controller
from extravaganza.utils import sample

"""
DEFINES A CLASS TO GENERATE RANDOM CONTROLS FOR THE PURPOSE OF EXPLORATION
"""

GEN_LENGTH = 2000  # number of random numbers to generate at any given time

class Explorer(Controller):
    def __init__(self, mean: np.ndarray, std: float,
                 sampling_method: str = 'normal', repeat_len: int = 5,
                 random_prob: float = 1.0, repeat_prob: float = 0.0, impulse_prob: float = 0.0):
        assert sampling_method in ['normal', 'sphere']
        self.control_dim = mean.shape[0]
        self.state_dim = -1
        self.mean = mean
        self.std = std
        self.sampling_method = sampling_method
        self.repeat_len = repeat_len
        
        probs = np.array([random_prob, repeat_prob, impulse_prob])
        assert np.allclose(probs.sum(), 1.), 'exploration sequence probabilities dont sum to 1?'
        self.keys = ['random', 'repeat', 'impulse']
        self.probs = probs
        
        self.controls = []
        self._generate_controls()
        self.stats = None
        pass
    
    def _generate_controls(self):
        logging.info('(EXPLORER) generating exploration control sequences using {} w.p. {}'.format(self.keys, self.probs))
        # get `GEN_LENGTH` random controls
        r = sample((GEN_LENGTH, self.control_dim), sampling_method=self.sampling_method)
        r = self.mean + r * self.std
        idx = 0
        
        # make the dataset of controls
        while idx + self.repeat_len <= GEN_LENGTH:
            key = np.random.choice(self.keys, p=self.probs)
            if key == 'random':
                pass
            elif key == 'repeat':
                r[idx + 1: idx + self.repeat_len] = r[idx]
            elif key == 'impulse':
                r[idx + 1: idx + self.repeat_len] = 0
            idx += self.repeat_len
        self.controls = list(r)
        pass
    
    def get_control(self, cost, state):
        if len(self.controls) == 0:
            self._generate_controls()
        return self.controls.pop()
    
if __name__ == '__main__':
    du = 5
    e = Explorer(mean=np.random.randn(5), std=1, sampling_method='normal', repeat_len=5, random_prob=0.7, repeat_prob=0.1, impulse_prob=0.2)
    for _ in range(4000):
        control = e.get_control(None, None)
        assert control.shape == (du,), 'failed :('
    print('passed :)')
    