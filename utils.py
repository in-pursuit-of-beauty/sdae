import os
import torch
import modules
import operator
import numpy as np
import torch.nn as nn
from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import RESISC45Dataset


def product(iterable):
    """Source: https://stackoverflow.com/a/595409."""
    return reduce(operator.mul, iterable, 1)


def to_img(x):
    if len(x.size()) < 4:
        h = w = int(np.sqrt(product(list(x.size())[1:])))
        x = x.view(x.size(0), 1, h, w)
    return x


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def zero_mask(x, zero_frac):
    """Apply zero-masking noise to a PyTorch tensor.
    Returns noisy X and a bitmask describing the affected locations."""
    bitmask = torch.rand_like(x) > zero_frac  # approx. ZERO_FRAC zeros
    return x * bitmask.float(), bitmask  # assumes the minimum value is 0


def add_gaussian(x, gaussian_stdev):
    """Apply isotropic additive Gaussian noise to a PyTorch tensor.
    Returns noisy X and a bitmask describing the affected locations."""
    noise = torch.empty_like(x).normal_(0, gaussian_stdev)
    return x + noise, torch.ones_like(x, dtype=torch.uint8)


def salt_and_pepper(x, sp_frac, minval=0.0, maxval=1.0):
    """Apply salt-and-pepper noise to a PyTorch tensor.
    Returns noisy X and a bitmask describing the affected locations."""
    rand = torch.rand_like(x)
    min_idxs = rand < (sp_frac / 2.0)
    max_idxs = rand > (1.0 - sp_frac / 2.0)
    x_sp = x.clone()
    x_sp[min_idxs] = minval
    x_sp[max_idxs] = maxval
    return x_sp, torch.clamp(min_idxs + max_idxs, 0, 1)


def save_image_wrapper(img, filepath):
    save_image(img, filepath)
    print('[o] saved image to %s' % filepath)


def init_model(model_class, restore_path, restore_required, **model_kwargs):
    # instantiate model
    model = getattr(modules, model_class)(**model_kwargs)
    if torch.cuda.is_available():
        model = model.cuda()
    print('instantiated a model of type %s' % model.__class__.__name__)
    # restore parameters
    if restore_required or restore_path:
        if restore_required or os.path.exists(restore_path):
            model.load_state_dict(torch.load(restore_path))
            print('restored "%s" model from %s' % (model_class, restore_path))
        else:
            print('warning: checkpoint %s not found, skipping...' % restore_path)
    return model


def init_loss(loss_type, **loss_kwargs):
    Loss = {
        'mse': nn.MSELoss,
        'bce': nn.BCELoss,
        'binary_cross_entropy': nn.BCELoss,
        'nll': nn.NLLLoss,
        'vae': modules.VAELoss,
    }[loss_type.lower()]
    print('using %r as the loss' % (Loss,))
    return Loss(**loss_kwargs)


def init_data_loader(dataset_key, batch_size=128, dataset_path=None):
    dataset_key = dataset_key.lower()
    if dataset_key.startswith('resisc'):
        # RESISC45
        dataset = RESISC45Dataset(dataset_path, normalize=False)
        c, h, w = dataset.get_data_dims()
    else:
        raise ValueError('unrecognized dataset: %s' % dataset_key)
    data_minval = dataset.get_minval()
    data_maxval = dataset.get_maxval()
    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    return data_loader, c, h, w, data_minval, data_maxval
