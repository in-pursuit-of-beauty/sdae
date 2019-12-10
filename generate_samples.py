import torch
import modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import init_model, init_data_loader


def plot_samples(samples, fig_save_path=None):
    """Given a tensor of samples
    [of shape (num_originals, num_variations+1, sh, sw)]
    corresponding to Figure 15 from the 2010 SDAE paper,
    plots the samples in a grid of variations as per Figure 15."""
    nrows, ncols = samples.shape[:2]
    fig, ax = plt.subplots(nrows, ncols)
    if nrows == 1 and ncols == 1:
        ax = [[ax]]
    elif nrows == 1:
        ax = [[col for col in ax]]
    elif ncols == 1:
        ax = [[row] for row in ax]
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.imshow(samples[i, j, :, :], cmap='gray')
            col.axis('off')
    if fig_save_path is not None:
        fig.savefig(fig_save_path)
        print('[o] saved figure to %s' % fig_save_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', type=str, default='CVAE')
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--fig_save_path', type=str, default=None)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--sample_h', type=int, default=256)
    parser.add_argument('--sample_w', type=int, default=256)
    parser.add_argument('--lower', type=float, default=-3)
    parser.add_argument('--upper', type=float, default=3)

    args = parser.parse_args()
    print(args)
    print('----------')

    model = init_model(args.model_class, args.restore_path, restore_required=True, latent_dim=512)
    model.eval()

    lower         = args.lower
    upper         = args.upper
    num           = args.num
    sample_h      = args.sample_h
    sample_w      = args.sample_w
    fig_save_path = args.fig_save_path

    # generate samples
    with torch.no_grad():
        # vary first two dims over grid
        dim0_vals = np.linspace(lower, upper, num)
        dim1_vals = np.linspace(lower, upper, num)
        dim0_vals, dim1_vals = np.meshgrid(dim0_vals, dim1_vals)
        latent_vecs = np.stack((dim0_vals, dim1_vals), axis=-1).reshape(-1, 2)

        latent_dim = model.latent_dim
        if latent_dim > 2:
            # keep values for other dimensions fixed
            other_vals = np.random.randn(latent_dim - 2)
            other_vals = np.expand_dims(other_vals, axis=0)  # shape: (1, ld-2)
            other_vals = np.tile(other_vals, (num * num, 1))  # shape: (n*n, ld-2)
            latent_vecs = np.concatenate((latent_vecs, other_vals), axis=1)  # shape: (n*n, ld)

        latent_vecs = torch.from_numpy(latent_vecs).float()
        if torch.cuda.is_available():
            latent_vecs = latent_vecs.cuda()
        samples = model.decode(latent_vecs)  # shape: (n*n, sample_h*sample_w)
        samples = samples.view(num, num, sample_h, sample_w)
        plot_samples(samples.cpu().numpy(), fig_save_path)
