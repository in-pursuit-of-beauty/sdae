import os
import torch
import imageio
import modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import init_model, init_data_loader


def plot_samples(samples, fig_save_path):
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


def encode_image(model, image_path):
    image_tensor = load_image_as_tensor(image_path)
    encoding, _, _ = model.encode(image_tensor)
    return np.squeeze(encoding.cpu().numpy())  # shape: (latent_dim,)


def generate_from_latent_vecs(model, latent_vecs):
    # Run generator.
    # `latent_vecs` should be a NumPy array of shape (batch, latent_dim).
    latent_vecs = torch.from_numpy(latent_vecs).float()
    if torch.cuda.is_available():
        latent_vecs = latent_vecs.cuda()
    samples = model.decode(latent_vecs)
    return samples.cpu().numpy()  # shape: (batch, sample_h*sample_w)


def load_image_as_tensor(image_path):
    image = imageio.imread(image_path, as_gray=True)
    while len(image.shape) < 4:
        image = np.expand_dims(image, 0)
    if image.max() > 1:
        image /= 255.0
    image = torch.from_numpy(image).float()
    if torch.cuda.is_available():
        image = image.cuda()
    return image


def autoencode_image(model, image_path, sample_h, sample_w):
    latent_vecs = encode_image(model, image_path)
    latent_vecs = np.expand_dims(latent_vecs, 0)
    image = generate_from_latent_vecs(model, latent_vecs)
    return image[0].reshape(sample_h, sample_w)  # (h, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', type=str, default='CVAE')
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--fig_save_path', type=str, default=None)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--sample_h', type=int, default=256)
    parser.add_argument('--sample_w', type=int, default=256)
    parser.add_argument('--lower', type=float, default=-3)
    parser.add_argument('--upper', type=float, default=3)
    parser.add_argument('--starter_im1_path', type=str)
    parser.add_argument('--starter_im2_path', type=str)

    args = parser.parse_args()
    print(args)
    print('----------')

    latent_dim = 512
    model = init_model(args.model_class, args.restore_path, restore_required=True, latent_dim=latent_dim)
    model.eval()

    lower         = args.lower
    upper         = args.upper
    num           = args.num
    sample_h      = args.sample_h
    sample_w      = args.sample_w
    fig_save_path = args.fig_save_path
    out_dir       = args.out_dir

    starter_im1_path = args.starter_im1_path
    starter_im2_path = args.starter_im2_path

    # generate samples
    with torch.no_grad():
        if out_dir:
            # save NUM samples to `out_dir`
            # first, generate latent vectors
            encoding1, encoding2 = None, None
            if starter_im1_path:
                encoding1 = encode_image(model, starter_im1_path)
            if starter_im2_path:
                encoding2 = encode_image(model, starter_im2_path)
            if encoding1 is not None and encoding2 is not None:
                # create latent vectors by interpolating between the encodings
                latent_vecs = np.array(
                    [(1.0 - w) * encoding1 + w * encoding2 for w in np.linspace(0, 1, num)])
            else:
                latent_vecs = np.random.normal(size=(num, latent_dim))  # shape: (n, latent)
                if encoding1 is not None:
                    # create latent vectors as random perturbations of the encoding
                    latent_vecs += np.expand_dims(encoding1, 0)

            # next, generate samples from latent vectors
            samples = generate_from_latent_vecs(model, latent_vecs)
            samples = samples.reshape(num, sample_h, sample_w)
            for i in range(num):
                out_path = os.path.join(out_dir, str(i).zfill(3) + '.png')
                imageio.imwrite(out_path,
                    (np.clip(samples[i], 0, 1) * 255).astype(np.uint8))
                print('Wrote `%s`.' % out_path)

        if fig_save_path:
            # vary first two dims over grid
            num = max(int(np.sqrt(num)), 1)
            dim0_vals = np.linspace(lower, upper, num)
            dim1_vals = np.linspace(lower, upper, num)
            dim0_vals, dim1_vals = np.meshgrid(dim0_vals, dim1_vals)
            latent_vecs = np.stack((dim0_vals, dim1_vals), axis=-1).reshape(-1, 2)

            if latent_dim > 2:
                # keep values for other dimensions fixed
                other_vals = np.random.randn(latent_dim - 2)
                other_vals = np.expand_dims(other_vals, axis=0)  # shape: (1, latent-2)
                other_vals = np.tile(other_vals, (num * num, 1))  # shape: (n*n, latent-2)
                latent_vecs = np.concatenate((latent_vecs, other_vals), axis=1)  # shape: (n*n, latent)

            samples = generate_from_latent_vecs(model, latent_vecs)  # shape: (n*n, sample_h*sample_w)
            samples = samples.reshape(num, num, sample_h, sample_w)
            plot_samples(samples, fig_save_path)
