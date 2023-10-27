import torch
import math

class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)

class Decoder(torch.nn.Module):

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
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super(Decoder, self).__init__()

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii+1]))

        self.layer_output = LipschitzLinear(dims[-2], dims[-1])
        self.tahn = torch.nn.Tanh()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x):
        xyz = x[:, -3:] * 100
        latent_vecs = x[:, :-3]
        x = torch.cat([latent_vecs, xyz], 1)
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.tahn(x)
        return self.layer_output(x)
