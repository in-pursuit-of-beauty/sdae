import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Source: https://bit.ly/2I8PJyH."""
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, c, h, w):
        super(Reshape, self).__init__()
        self.shape = (c, h, w)

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


# ========================
# VARIATIONAL AUTOENCODERS
# ========================


class CVAE(nn.Module):
    """Convolutional variational autoencoder.
    Based on https://www.tensorflow.org/tutorials/generative/cvae.
    """

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.is_convolutional = True

        # shape: (batch, 1, 256, 256)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),  # shape: (batch, 16, 126, 126)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),  # shape: (batch, 32, 61, 61)
            Flatten(),  # shape: (batch, 32 * 61 * 61)
            nn.Linear(32 * 61 * 61, latent_dim * 2),
        )
        self.mean_estimator = nn.Linear(
            in_features=latent_dim*2, out_features=latent_dim)
        self.log_var_estimator = nn.Linear(
            in_features=latent_dim*2, out_features=latent_dim)
        # shape: (batch, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * 64 * 64),
            nn.ReLU(),
            Reshape(16, 64, 64),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=6, stride=1, padding=1),
            nn.Sigmoid(),
        )
        # shape: (batch, 1, 256, 256)

    def forward(self, x):
        z, mean, log_var = self.encode(x)
        return self.decode(z), mean, log_var

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_estimator(x)
        log_var = self.log_var_estimator(x)
        x = self.sample_latent_vector(mean, log_var)
        return x, mean, log_var

    @staticmethod
    def sample_latent_vector(mean, log_var):
        # ------------------------
        # reparameterization trick
        # ------------------------
        # MEAN and LOG_VAR are "mu" and log("sigma" ^2) using
        # the notation from eq. 10 in Auto-Encoding Variational Bayes
        # -----------------------------------------------------------
        stdev = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(stdev)
        return mean + stdev * epsilon  # sampled latent vector

    def decode(self, x):
        return self.decoder(x)


class VAELoss(nn.Module):
    def __init__(self, reduction='sum', reconstruction_loss_type='mse'):
        super(VAELoss, self).__init__()
        self.reduction = reduction
        self.reconstruction_loss_type = reconstruction_loss_type.lower()
        assert self.reconstruction_loss_type in {'mse', 'bce', 'binary_cross_entropy'}

    def forward(self, input_, target, mean, log_var):
        # reconstruction
        if self.reconstruction_loss_type == 'mse':
            reconstruction_loss = F.mse_loss(input_, target, reduction=self.reduction)
        else:
            reconstruction_loss = F.binary_cross_entropy(input_, target, reduction=self.reduction)

        # regularization
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        kl_div = 0.5 * torch.sum(mean * mean + torch.exp(log_var) - log_var - 1, dim=1)
        kl_div = torch.mean(kl_div) if self.reduction == 'mean' else torch.sum(kl_div)

        return reconstruction_loss + kl_div
