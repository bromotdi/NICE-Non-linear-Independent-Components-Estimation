"""NICE model"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform ,SigmoidTransform, AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np

"""Additive coupling layer"""
class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()

        self.mask_config = mask_config
       
        self.input_layer = nn.Sequential(
            nn.Linear(in_out_dim // 2, mid_dim),
            nn.ReLU()
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()
            ) 
            for _ in range(hidden - 1)]
        )

        self.output_layer = nn.Linear(mid_dim, in_out_dim // 2)
        

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        # reshape x to match NICE shapes.
        size0, size1 = x.size()

        x = x.reshape((x.shape[0], x.shape[1] // 2, 2))

        # determine units to transform on using mask-config.
        x1, x2 = x[:, :, 1], x[:, :, 0]
        if self.mask_config:
            x1, x2 = x[:, :, 0], x[:, :, 1]

        # transform half of the data using the predefines model
        out = self.input_layer(x2)
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
        out = self.output_layer(out)

        # apply additive function
        if reverse:
            x1 = x1 - out
        else:
            x1 = x1 + out

        # return x with proper stack
        x = torch.stack((x2, x1), dim=2)
        if self.mask_config:
            x = torch.stack((x1, x2), dim=2)

        out_param = x.reshape((size0, size1)), log_det_J
        return out_param

class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        self.mask_config = mask_config

        self.input_layer = nn.Sequential(
            nn.Linear(in_out_dim // 2, mid_dim),
            nn.ReLU()
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()
            ) 
            for _ in range(hidden - 1)]
        )

        self.output_layer = nn.Linear(mid_dim, in_out_dim)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        if self.mask_config:
            x1, x2 = (x[:, 1::2], x[:, 0::2])
        else:
            x1, x2 = (x[:, 0::2], x[:, 1::2])

        out = self.input_layer(x2)
        
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
            
        out = self.output_layer(out)

        log_s, t = out[:, 0::2, ...], out[:, 1::2, ...]
        s = torch.sigmoid(log_s)

        if reverse:
            x1 = (x1 - t) / s
        else:
            x1 = s * x1 + t

        log_det_J = torch.sum(torch.log(torch.abs(s)))

        x = torch.ones_like(x)
        if self.mask_config:
            x[:, 1::2], x[:, 0::2] = (x1, x2)
        else:
            x[:, 1::2], x[:, 0::2] = (x2, x1)

        return x, log_det_J

"""
    Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale)+ self.eps
        log_det_J = torch.sum(self.scale)

        if reverse:
            x = x * (scale ** -1)
        else:
            x = x * scale

        return x, log_det_J

"""
    Standard logistic distribution
"""
logistic = TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(loc=0., scale=1.)])

"""
    NICE main model
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type, in_out_dim, mid_dim, hidden, device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        
        self.device = device
        
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
            
        elif prior == 'logistic':
            self.prior = logistic
            
        else:
            raise ValueError('Prior not implemented.')
        
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.scaling = Scaling(self.in_out_dim)
        
        if self.coupling_type == "additive":
            self.coupling = nn.ModuleList([
                AdditiveCoupling(in_out_dim=self.in_out_dim,
                                 mid_dim=self.mid_dim,
                                 hidden=self.hidden,
                                 mask_config=i % 2) 
                for i in range(coupling)
            ])

        elif self.coupling_type == 'affine':
            self.coupling = nn.ModuleList([
                AffineCoupling(in_out_dim=self.in_out_dim,
                               mid_dim=self.mid_dim,
                               hidden=self.hidden,
                               mask_config=i % 2) 
                for i in range(coupling)
            ])

        else:
            raise ValueError('Coupling type not implemented.')

    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x, _ = self.coupling[i](x, 0, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        
        log_det_j = 0
    
        for i in range(len(self.coupling)):
            x, ldj = self.coupling[i](x, log_det_j)
            log_det_j += ldj

        x, ldj = self.scaling(x, reverse=False)
        log_det_j += ldj

        return x, log_det_j

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256)*self.in_out_dim #log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)