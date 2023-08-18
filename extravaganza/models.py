import logging
from typing import List, Iterable

from extravaganza.utils import set_seed, jkey, least_squares, method_of_moments

#  ============================================================================================
#  ============================================================================================
#  ===========================================  JAX  ==========================================
#  ============================================================================================
#  ============================================================================================

import jax.numpy as jnp
import flax.linen as jnn

class JaxMLP(jnn.Module):
    """
    made using the tutorial found at:
        https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
    """
    
    layer_dims: List[int]   # dims of each layer
    activation: jnn.Module = jnn.activation.relu
    normalization: jnn.Module = None  # normalize before the activation
    drop_last_activation: bool = True
    use_bias: bool = True
    seed: int = None
        
    def setup(self):  # create the modules to build the network
        set_seed(self.seed)
        
        self.input_dim = self.layer_dims[0]
        self.output_dim = self.layer_dims[-1]
        
        layers = []

        for i in range(len(self.layer_dims) - 1):
            in_dim, out_dim = self.layer_dims[i: i + 2]
            layers.append(jnn.Dense(features=out_dim, use_bias=self.use_bias))
            if self.normalization is not None: layers.append(self.normalization())
            layers.append(self.activation)
        if self.drop_last_activation: 
            layers.pop()  # removes activation from final layer
            if self.normalization is not None: layers.pop()  # removes normalization too
        
        self.model = jnn.Sequential(layers)
        pass
    
    def __call__(self, x: jnp.ndarray):  # forward pass
        x = x.reshape(-1, self.layer_dims[0])
        x = self.model(x)
        return x
    
def get_jax_mlp(layer_dims: List[int],
                activation: jnn.Module = jnn.activation.relu,
                normalization: jnn.Module = None,  # normalize before the activation
                drop_last_activation: bool = True,
                use_bias: bool = True,
                seed: int = None):
    
    model = JaxMLP(layer_dims=layer_dims, activation=activation, normalization=normalization, drop_last_activation=drop_last_activation, use_bias=use_bias, seed=seed)
    params = model.init(jkey(), jnp.zeros((1, layer_dims[0])))
    return model, params



#  ============================================================================================
#  ============================================================================================
#  =========================================  PYTORCH  ========================================
#  ============================================================================================
#  ============================================================================================

import torch
import torch.nn as nn
from info_nce import InfoNCE
from pytorch_metric_learning.losses import ContrastiveLoss, SelfSupervisedLoss, SupConLoss

class TorchMLP(nn.Module):
    def __init__(self, 
                 layer_dims: List[int], 
                 activation: nn.Module = nn.ReLU,
                 normalization: nn.Module = nn.Identity,  # normalize before the activation
                 drop_last_activation: bool = True,
                 use_bias: bool = True,
                 seed: int = None):
        """
        Creates a TorchMLP to use as a weak learner

        Parameters
        ----------
        layer_dims : List[int]
            dimensions of each layer (`layer_dims[0]` should be input dim and `layer_dims[-1]` should be output dim)
        activation : nn.Module, optional
            activation function, by default nn.ReLU
        """
        
        super(TorchMLP, self).__init__()
        set_seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.use_bias = use_bias
        
        self.layers = [nn.Flatten(1)]
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i: i + 2]
            self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            self.layers.append(normalization(out_dim))
            self.layers.append(activation())
        if drop_last_activation: self.layers.pop()  # removes activation from final layer
        
        self.model = nn.Sequential(*self.layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model(x)
        return h
    
    
class TorchCNN(nn.Module):
    def __init__(self, 
                 input_shape: Iterable,  # should be in CxHxW
                 output_dim: int,
                 activation: nn.Module=nn.ReLU,
                 use_bias=True,
                 seed: int = None):
        super(TorchCNN, self).__init__()
        set_seed(seed)
        
        assert len(input_shape) in [2, 3]
        if len(input_shape) == 2: input_shape = (1, *input_shape)
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        self.body = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3),
            activation(),
            # nn.Conv2d(16, 32, kernel_size=3),
            # activation(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size(), 32, bias=use_bias),
            activation(),
            nn.Linear(32, self.output_dim, bias=use_bias)
        )
        
    def forward(self, x) -> torch.Tensor:
        x = x.reshape(-1, *self.input_shape)
        h = self.body(x)
        h = self.fc(h)
        return h
    
    def feature_size(self) -> int:
        with torch.no_grad():
            feature_size = self.body(torch.zeros(1, *self.input_shape)).view(1, -1).shape[1]
        return feature_size

class TorchAutoencoder(nn.Module):
    def __init__(self, 
                 layer_dims: List[int], 
                 activation: nn.Module = nn.ReLU,
                 normalization: nn.Module = nn.Identity,  # normalize before the activation
                 drop_last_activation: bool = True,
                 use_bias: bool = True,
                 seed: int = None):
        super().__init__()
        self.encoder = TorchMLP(layer_dims=layer_dims, 
                                activation=activation, 
                                normalization=normalization, 
                                drop_last_activation=drop_last_activation, 
                                use_bias=use_bias, 
                                seed=seed)
        self.decoder = TorchMLP(layer_dims=layer_dims[::-1],
                                activation=activation,
                                normalization=nn.Identity,
                                drop_last_activation=True,
                                use_bias=True,
                                seed=seed)
        # import logging; logging.warning('GUYS IM DOING THE SPHERE * COST EMBEDDING!!!! WATCH OUT!!! :)')
        pass
    
    def encode(self, x): 
        emb = self.encoder(x)
        # emb = emb / torch.linalg.norm(emb, dim=-1).unsqueeze(dim=-1)  # proj to unit sphere
        # emb = emb * 50 * torch.abs(x[:, 2]).unsqueeze(dim=-1)  # proj to cost sphere
        return emb
    
    def decode(self, z): return self.decoder(z)
    
class TorchPC3(nn.Module):
    def __init__(self, 
                 encoder: TorchMLP,
                 x_dim: int,
                 u_dim: int,
                 z_dim: int,
                 AB_method: str = 'learned',
                 sigma: float = 0,
                 determinstic_encoder: bool = False,
                 decoder: TorchMLP = None, 
                 do_cpc: bool = False,
                 do_jac: bool = False):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert encoder.input_dim == x_dim, 'inconsistent dims'
        if decoder is not None: 
            assert encoder.output_dim == decoder.input_dim and decoder.output_dim == x_dim, 'inconsistent dims'
            logging.info('(PC3): decoder provided, so reconstruction error WILL be computed')
        else:
            logging.info('(PC3): decoder not provided, so reconstruction error will NOT be computed')
        
        self.obs_dim, self.state_dim, self.control_dim = x_dim, z_dim, u_dim
        self.sigma = sigma
        
        self.do_cpc = do_cpc
        if self.do_cpc: 
            """
            Contrastive predictive coding, see https://arxiv.org/pdf/1807.03748.pdf.
            """
            self.cpc_loss = InfoNCE(negative_mode='unpaired')  # 'paired' means each anchor gets compared to only its negative, instead of all negatives (for that, change to 'unpaired')
            # self.cpc_loss = SelfSupervisedLoss(ContrastiveLoss(), symmetric=False)
            logging.info('(PC3): contrastive predictive coding WILL be computed')
        else: 
            logging.info('(PC3): contrastive predictive coding will NOT be computed')
            
        self.do_jac = do_jac
        if self.do_jac: logging.info('(PC3): jacobian loss WILL be computed')
        else: logging.info('(PC3): jacobian loss will NOT be computed')
        
        self.mu = nn.Linear(self.encoder.output_dim, z_dim)
        self.logvar = nn.Linear(self.encoder.output_dim, z_dim) if not determinstic_encoder else None
        
        if self.decoder is not None:
            # train the decoder to match the encoder in the beginning
            num_iters, batch_size = 5000, 256
            decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=0.002)
            losses = []
            for _ in range(num_iters):
                decoder_opt.zero_grad()
                x = torch.randn((batch_size, x_dim))
                z = self.encode(x)
                loss = torch.nn.functional.mse_loss(self.decoder(z), x)
                loss.backward()
                decoder_opt.step()
                losses.append(loss.item())
            import matplotlib.pyplot as plt
            plt.plot(range(len(losses)), losses)
            plt.show()
            
        assert AB_method in ['learned', 'regression_nograd', 'regression', 'moments_nograd', 'moments']
        logging.info('(PC3): using "{}" method to get the AB matrices during each training step'.format(AB_method))
        if AB_method == 'learned':
            # self.A = torch.nn.Parameter(torch.ones(z_dim, dtype=torch.float32, requires_grad=True))
            self.A = torch.nn.Parameter(torch.eye(z_dim, dtype=torch.float32, requires_grad=True))
            self.B = torch.nn.Parameter(torch.randn((z_dim, u_dim), dtype=torch.float32, requires_grad=True))
            # self.get_AB = lambda xs, us, mask: (torch.diag(self.A), self.B)
            self.get_AB = lambda xs, us, mask: (self.A, self.B)
        elif AB_method == 'regression_nograd':
            def get_AB(xs, us, mask):
                with torch.no_grad(): return least_squares(xs, us, mask=mask)
            self.get_AB = get_AB
        elif AB_method == 'regression': self.get_AB = least_squares
        elif AB_method == 'moments_nograd':
            def get_AB(xs, us, mask):
                with torch.no_grad(): return method_of_moments(xs, us, mask=mask)
            self.get_AB = get_AB
        elif AB_method == 'moments': self.get_AB = method_of_moments
        else: raise NotImplementedError(AB_method)
        pass
        
    def encode(self, x: torch.Tensor): 
        unbatch = x.ndim == 1
        if unbatch: x = x.unsqueeze(0)
        assert x.shape[1] == self.obs_dim, (x.shape, self.obs_dim)
        z = self.encoder(x)
        enc = self.mu(z)
        if self.logvar is not None:  # actually sample from trained distribution
            logvar = self.logvar(z)
            enc = enc + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        assert enc.shape[1] == self.state_dim, (enc.shape, self.state_dim)
        if unbatch: enc = enc.squeeze(0)
        return enc
    
    def get_embs_and_losses(self, obs, controls, mask = None): 
        """
        use `prev_mask` for loss terms indexed on the x_t side.
        use `next_mask` for loss terms indexed on the x_{t+1} side.
        """
        assert controls.ndim == 2 and controls.shape[1] == self.control_dim, (controls.shape, self.control_dim)
        assert obs.shape[0] == controls.shape[0], (obs.shape[0], controls.shape[0])
        
        if mask is not None: assert mask.shape == (obs.shape[0] - 1,), (mask.shape, obs.shape[0] - 1)
        else: mask = torch.ones(obs.shape[0] - 1, dtype=bool)

        z = self.encode(obs)
        zprev = z[:-1]
        zgt = z[1:]
        
        # estimate linear predicted embeddings
        A, B = self.get_AB(z, controls, mask)
        zhat = (A @ zprev.unsqueeze(-1) + B @ controls[:-1].unsqueeze(-1)).squeeze(-1)
        
        # perturb the g.t. next state encoding, see Section 4.2 of PC3 paper
        if self.sigma > 0: zgt = zgt + self.sigma * torch.randn_like(zgt)
        
        losses = {}
        if self.do_cpc: losses['cpc'] = self.cpc_loss(zhat[mask], zgt[mask], None)  # contrastive predictive coding -- we want z to be close to zhat and far from zprev
        if self.decoder is not None: losses['reconstruction'] = torch.nn.functional.mse_loss(self.decoder(z), obs)
        if self.do_jac:
            jacs = torch.func.vmap(torch.func.jacrev(self.encode))(obs[:-1])
            LHS = (jacs @ (obs[1:] - obs[:-1]).unsqueeze(-1)).squeeze(-1)
            RHS = (A @ zprev.unsqueeze(-1) + B @ controls[:-1].unsqueeze(-1)).squeeze(-1) - zprev
            losses['jac'] = torch.nn.functional.mse_loss(LHS[mask], RHS[mask])
        
        return (z, zhat), (A, B), losses
    

if __name__ == '__main__':
    print('testing model dimension stuff!')
    _TorchMLP = TorchMLP([128, 256, 128, 64, 16])
    _TorchMLP_test = torch.zeros((1, 128))
    assert _TorchMLP(_TorchMLP_test).shape[1] == 16
    
    _TorchCNN = TorchCNN(input_shape=(3, 210, 160), output_dim=16)
    _TorchCNN_test = torch.zeros((1, 3, 210, 160))
    assert _TorchCNN(_TorchCNN_test).shape[1] == 16
    print('yippee!')
    
    from testing.utils import count_parameters
    TorchMLP = TorchMLP(layer_dims=[int(28 * 28), 100, 100, 100, 100, 10]).float()
    TorchCNN = TorchCNN(input_shape=(28, 28), output_dim=10)
    print(count_parameters(TorchMLP), count_parameters(TorchCNN))

