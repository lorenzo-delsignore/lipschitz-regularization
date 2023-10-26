#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class CustomLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(CustomLinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        #self.linear.weight = nn.Parameter(torch.tensor(np.random.randn(out_features, in_features) * np.sqrt(2 / in_features), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.c = nn.Parameter(torch.max(torch.sum(torch.abs(self.linear.weight), axis=1)), requires_grad= True) # Initialize 'c' as a trainable parameter
        self.activation = activation
        if self.activation == True:
            self.tanh = nn.Tanh()
    def forward(self, x):
        # Add the custom parameter 'c' to the output of the linear layer
        lipschitz_constant = self.c
        self.linear.weight.data = self.weight_normalization(self.linear.weight, F.softplus(lipschitz_constant))
        if self.activation == True:
            return self.tanh(self.linear(x))
        else:
            return self.linear(x)


    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(torch.tensor(1.0), softplus_c / absrowsum)
        return W * scale[:, None]

def get_lipschitz_loss(decoder):
    """
    This function computes the Lipschitz regularization
    """
    loss_lip = 1.0
    for name, layer in decoder.named_modules():
        if isinstance(layer, CustomLinearLayer):
            c = layer.c
            loss_lip = loss_lip * F.softplus(c)

    return loss_lip


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if layer == self.num_layers - 2:
                setattr(self, "lin" + str(layer), CustomLinearLayer(dims[layer], out_dim, activation=False))
            else:
                setattr(self, "lin" + str(layer), CustomLinearLayer(dims[layer], out_dim, activation=True))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.dropout_prob = dropout_prob
        self.dropout = dropout


    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:] * 100

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            latent_vecs = input[:, :-3]
            x = torch.cat([latent_vecs, xyz], 1)

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)

            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


    def forward_single(self, params_net, t, x):
        """
        Forward pass of a lipschitz MLP

        Inputs
        params_net: parameters of the network
        t: the input feature of the shape
        x: a query location in the space

        Outputs
        out: implicit function value at x
        """
        # forward pass
        for ii in range(len(params_net) - 1):
            W, b, c = params_net[ii]
            W = self.weight_normalization(W, F.softplus(c))
            x = self.hyperParams['activation_fn'](np.dot(W, x) + b)

        # final layer
        W, b, c = params_net[-1]
        W = self.weight_normalization(W, F.softplus(c))
        out = np.dot(W, x) + b
        return out[0]




    def normalize_params(self, params_net):
        """
        (Optional) After training, this function will clip network [W, b] based on learned lipschitz constants. Thus, one can use normal MLP forward pass during test time, which is a little bit faster.
        """
        params_final = []
        for ii in range(len(params_net)):
            W, b, c = params_net[ii]
            W = self.weight_normalization(W, torch.nn.Softplus(c))
            params_final.append([W, b])
        return params_final

    def forward_eval_single(self, params_final, t, x):
        """
        (Optional) this is a standard forward pass of a mlp. This is useful to speed up the performance during test time
        """
        # concatenate coordinate and latent code
        x = np.append(x, t)

        # forward pass
        for ii in range(len(params_final) - 1):
            W, b = params_final[ii]
            x = self.hyperParams['activation_fn'](np.dot(W, x) + b)
        W, b = params_final[-1]  # final layer
        out = np.dot(W, x) + b
        return out[0]

