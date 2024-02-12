import pandas as pd
import numpy as np
import matplotlib as plt
import collections

import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader



# The (A)NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

  Supports vector inputs (x) and vector outputs (y). Kernel is
  mean-squared exponential, using the x-value l2 coordinate distance scaled by
  some factor chosen randomly in a range. Outputs are independent gaussian
  processes.
      """

    def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               l1_scale=0.6,
               sigma_scale=1.0,
               random_kernel_parameters=True,
               testing=False):
        """Creates a regression dataset of functions sampled from a GP.

    Args:
      batch_size: An integer.
      max_num_context: The max number of observations in the context.
      x_size: Integer >= 1 for length of "x values" vector.
      y_size: Integer >= 1 for length of "y values" vector.
      l1_scale: Float; typical scale for kernel distance function.
      sigma_scale: Float; typical scale for variance.
      random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma) 
          will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
      testing: Boolean that indicates whether we are testing. If so there are
          more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

    Args:
      xdata: Tensor of shape [B, num_total_points, x_size] with
          the values of the x-axis data.
      l1: Tensor of shape [B, y_size, x_size], the scale
          parameter of the Gaussian kernel.
      sigma_f: Tensor of shape [B, y_size], the magnitude
          of the std.
      sigma_noise: Float, std of the noise that we add for stability.

    Returns:
      The kernel, a float tensor of shape
      [B, y_size, num_total_points, num_total_points].
        """
        num_total_points = xdata.shape[1]

    # Expand and take the difference
        xdata1 = xdata.unsqueeze(1)  # [B, 1, num_total_points, x_size]
        xdata2 = xdata.unsqueeze(2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

    # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :])**2

        norm = torch.sum(norm, -1)  # [B, data_size, num_total_points, num_total_points]

    # [B, y_size, num_total_points, num_total_points]
        kernel = ((sigma_f)**2)[:, :, None, None] * torch.exp(-0.5 * norm)

    # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * torch.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

    Generated functions are `float32` with x values between -2 and 2.
    
    Returns:
      A `CNPRegressionDescription` namedtuple.
        """
        num_context = int(np.random.rand()*(self._max_num_context - 3) + 3)
    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = torch.arange(-2, 2, 1.0/100).unsqueeze(0).repeat(self._batch_size, 1)
            x_values = x_values.unsqueeze(-1)
    # During training the number of target points and their x-positions are
    # selected at random
        else:
            num_target = int(np.random.rand()*(self._max_num_context - num_context))
            num_total_points = num_context + num_target
            x_values = torch.rand((self._batch_size, num_total_points, self._x_size))*4 - 2
            

    # Set kernel parameters
    # Either choose a set of random parameters for the mini-batch
        if self._random_kernel_parameters:
            l1 = torch.rand((self._batch_size, self._y_size, self._x_size))*(self._l1_scale - 0.1) + 0.1
            sigma_f = torch.rand((self._batch_size, self._y_size))*(self._sigma_scale - 0.1) + 0.1
            
    # Or use the same fixed parameters for all mini-batches
        else:
            l1 = torch.ones((self._batch_size, self._y_size, self._x_size))*self._l1_scale
            sigma_f = torch.ones((self._batch_size, self._y_size))*self._sigma_scale

    # Pass the x_values through the Gaussian kernel
    # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

    # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel)

    # Sample a curve
    # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(cholesky, torch.randn((self._batch_size, self._y_size, num_total_points, 1)))

    # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(3)
        y_values = y_values.permute(0, 2, 1)

        if self._testing:
      # Select the targets
            target_x = x_values
            target_y = y_values

      # Select the observations
            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context]]
            context_y = y_values[:, idx[:num_context]]

        else:
      # Select the targets which will consist of the context points as well as
      # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

      # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=target_x.shape[1],
        num_context_points=num_context)
    
class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_type, n_heads=8):
        super().__init__()
        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W = nn.Linear(n_heads*hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        else:
            raise NotImplementedError
            
    def forward(self, k, v, q):
        rep = self._attention_func(k, v, q)
        return rep
    
    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep
    
    def _laplace_attention(self, k, v, q, scale=0.5):
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(2)
        unnorm_weights = torch.abs((k_ - v_)*scale)
        unnorm_weights = unnorm_weights.sum(dim=-1)
        weights = torch.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum('bik,bkj->bij', weights, v)
        return rep
    
    def _dot_attention(self, k, v, q):
        scale = q.shape[-1]**0.5
        unnorm_weights = torch.einsum('bjk,bik->bij', k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)
        
        rep = torch.einsum('bik,bkj->bij', weights, v)
        return rep
    
    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_type, n_heads=8):
        super().__init__()
        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W = nn.Linear(n_heads*hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        else:
            raise NotImplementedError
            
    def forward(self, k, v, q):
        rep = self._attention_func(k, v, q)
        return rep
    
    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep
    
    def _laplace_attention(self, k, v, q, scale=0.5):
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(2)
        unnorm_weights = torch.abs((k_ - v_)*scale)
        unnorm_weights = unnorm_weights.sum(dim=-1)
        weights = torch.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum('bik,bkj->bij', weights, v)
        return rep
    
    def _dot_attention(self, k, v, q):
        scale = q.shape[-1]**0.5
        unnorm_weights = torch.einsum('bjk,bik->bij', k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)
        
        rep = torch.einsum('bik,bkj->bij', weights, v)
        return rep
    
    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep
   
class LatentEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=32, latent_dim=32, self_attention_type="dot", n_encoder_layers=3):
        super(LatentEncoder, self).__init__()
        self._input_layer = nn.Linear(input_dim, hidden_dim)
        self._encoder = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_encoder_layers)])
        self._self_attention = Attention(hidden_dim, self_attention_type)
        self._penultimate_layer = nn.Linear(hidden_dim, hidden_dim)
        self._mean = nn.Linear(hidden_dim, latent_dim)
        self._log_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, y):
        encoder_input = torch.cat([x,y], dim=-1)
        
        encoded = self._input_layer(encoder_input)
        
        for layer in self._encoder:
            encoded = torch.relu(layer(encoded))

        attention_output = self._self_attention(encoded, encoded, encoded)
        
        mean_repr = attention_output.mean(dim=1)
        
        mean_repr = torch.relu(self._penultimate_layer(mean_repr))
        
        mean = self._mean(mean_repr)
        log_var = self._log_var(mean_repr)
        
        z = torch.randn_like(log_var)*torch.exp(log_var/2.0) + mean
        
        return z, mean, log_var

class DeterministicEncoder(nn.Module):
    
    def __init__(self, input_dim, x_dim, hidden_dim=32, n_d_encoder_layers=3, self_attention_type="dot",
                 cross_attention_type="dot"):
        super(DeterministicEncoder, self).__init__()
        self._input_layer = nn.Linear(input_dim, hidden_dim)
        self._d_encoder = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_d_encoder_layers)])
        self._self_attention = Attention(hidden_dim, self_attention_type)
        self._cross_attention = Attention(hidden_dim, cross_attention_type)
        self._target_transform = nn.Linear(x_dim, hidden_dim)
        self._context_transform = nn.Linear(x_dim, hidden_dim)
    
    def forward(self, context_x, context_y, target_x):
        d_encoder_input = torch.cat([context_x,context_y], dim=-1)
        d_encoded = self._input_layer(d_encoder_input)
        for layer in self._d_encoder:
            d_encoded = torch.relu(layer(d_encoded))
        attention_output = self._self_attention(d_encoded, d_encoded, d_encoded)
        
        q = self._target_transform(target_x)
        k = self._context_transform(context_x)
        q = self._cross_attention(k, d_encoded, q)
        
        return q
    
class Decoder(nn.Module):
    
    def __init__(self, x_dim, y_dim, hidden_dim=32, latent_dim=32, n_decoder_layers=3):
        super(Decoder, self).__init__()
        self._target_transform = nn.Linear(x_dim, hidden_dim)
        self._decoder = nn.ModuleList([nn.Linear(2*hidden_dim+latent_dim, 2*hidden_dim+latent_dim) for _ in range(n_decoder_layers)])
        self._mean = nn.Linear(2*hidden_dim+latent_dim, y_dim)
        self._std = nn.Linear(2*hidden_dim+latent_dim, y_dim)
        self._softplus = nn.Softplus()
    def forward(self, r, z, target_x):
        x = self._target_transform(target_x)
        
        representation = torch.cat([torch.cat([r, z], dim=-1), x], dim=-1)
        for layer in self._decoder:
            representation = torch.relu(layer(representation))
            
        mean = self._mean(representation)
        std = self._softplus(self._std(representation))
        pred = torch.randn_like(std)*std + mean
        return pred, std

class LatentModel(nn.Module):
    
    def __init__(self, x_dim, y_dim, hidden_dim=32, latent_dim=32, latent_enc_self_attn_type="dot",
                det_enc_self_attn_type="dot", det_enc_cross_attn_type="dot", n_latent_encoder_layers=3,
                n_det_encoder_layers=3, n_decoder_layers=3):
        
        super(LatentModel, self).__init__()
        
        self._latent_encoder = LatentEncoder(x_dim + y_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, 
                                             self_attention_type=latent_enc_self_attn_type,
                                             n_encoder_layers=n_latent_encoder_layers)
        
        self._deterministic_encoder = DeterministicEncoder(x_dim + y_dim, x_dim, hidden_dim=hidden_dim,
                                                           self_attention_type=det_enc_self_attn_type,
                                                           cross_attention_type=det_enc_cross_attn_type,
                                                           n_d_encoder_layers=n_det_encoder_layers)
        
        self._decoder = Decoder(x_dim, y_dim, hidden_dim=hidden_dim, n_decoder_layers=n_decoder_layers)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)
        
        z_prior, mean_prior, log_var_prior = self._latent_encoder(context_x, context_y)
        
        if target_y is not None:
            z_post, mean_post, log_var_post = self._latent_encoder(target_x, target_y)
            z = z_post
        else:
            z = z_prior
        
        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]
        r = self._deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]
        pred, y_std = self._decoder(r, z, target_x)
        if target_y is not None:
            mse_loss = self.mse_loss(pred, target_y)
            kl_loss = self.kl_loss(mean_prior, log_var_prior, mean_post, log_var_post)
            loss = mse_loss + kl_loss
        else:
            mse_loss = None
            kl_loss = None
            loss = None
        
        return pred, kl_loss, loss, y_std
    
    def kl_loss(self, mean_prior, log_var_prior, mean_post, log_var_post):
        kl_loss = 0.5*((torch.exp(log_var_post) + (mean_post - mean_prior)**2)/torch.exp(log_var_prior) - 1. + \
        (log_var_prior - log_var_post)).sum()
        return kl_loss


def get_fitted_model_anp(dataset_train, dataset_test, state_dict=None, optim=None):
    # initialize and fit model
    model = LatentModel(1, 1, 64, latent_enc_self_attn_type="multihead", det_enc_self_attn_type="multihead",
                   det_enc_cross_attn_type="multihead")
    if state_dict is not None:
        model.load_state_dict(state_dict)

    if optim == None:
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        optim = optim

    epochs = 10000
    for epoch in range(epochs):
        model.train()
        data_train = dataset_train.generate_curves()
        (context_x, context_y), target_x = data_train.query
        target_y = data_train.target_y
        context_x = context_x.cuda()
        context_y = context_y.cuda()
        target_x = target_x.cuda()
        target_y = target_y.cuda()
        optim.zero_grad()
        y_pred, kl, loss, y_std = model(context_x, context_y, target_x, target_y)
        if epoch % 100 == 0:
            print("train: ", epoch, loss.item())
            loss.backward()
            optim.step()
        
        # if epoch % 1000 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         data_test = dataset_test.generate_curves()
        #         (context_x, context_y), target_x = data_test.query
        #         target_y = data_test.target_y
        #         context_x = context_x.cuda()
        #         context_y = context_y.cuda()
        #         target_x = target_x.cuda()
        #         target_y = target_y.cuda()
        #         y_pred, kl, loss, y_std = model(context_x, context_y, target_x)
        #         plot_functions(target_x.detach().cpu().numpy(),
        #                        target_y.detach().cpu().numpy(),
        #                        context_x.detach().cpu().numpy(),
        #                        context_y.detach().cpu().numpy(),
        #                        y_pred.detach().cpu().numpy(),
        #                        y_std.detach().cpu().numpy())

    return model