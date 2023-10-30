import torch

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

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(torch.nn.Linear(dims[ii], dims[ii+1]))

        self.layer_output = torch.nn.Linear(dims[-2], dims[-1])
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        xyz = x[:, -3:] * 100
        latent_vecs = x[:, :-3]
        x = torch.cat([latent_vecs, xyz], 1)
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.tanh(x)
        return self.layer_output(x)
