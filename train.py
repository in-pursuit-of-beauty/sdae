"""
Denoising variational autoencoder.
Code originally based on https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder.
"""

import os
import torch
import modules
import argparse
import numpy as np
from utils import to_img, zero_mask, add_gaussian, salt_and_pepper, \
    save_image_wrapper, init_model, init_loss, init_data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_class', type=str, default='CVAE')
    parser.add_argument('--dataset_key', type=str, default='resisc')
    parser.add_argument('--noise_type', type=str, default='gs')
    parser.add_argument('--zero_frac', type=float, default=0.3)
    parser.add_argument('--gaussian_stdev', type=float, default=0.4)
    parser.add_argument('--sp_frac', type=float, default=0.1)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='./model.pth')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--dataset_path', type=str, default='data/NWPU-RESISC45')
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()
    print(args)
    print('----------')

    batch_size     = args.batch_size
    learning_rate  = args.learning_rate
    num_epochs     = args.num_epochs
    model_class    = args.model_class
    dataset_key    = args.dataset_key
    noise_type     = args.noise_type
    zero_frac      = args.zero_frac
    gaussian_stdev = args.gaussian_stdev
    sp_frac        = args.sp_frac
    restore_path   = args.restore_path
    save_path      = args.save_path
    log_freq       = args.log_freq
    dataset_path   = args.dataset_path
    weight_decay   = args.weight_decay

    # set up log folders
    if not os.path.exists('./01_original'):
        os.makedirs('./01_original')
    if not os.path.exists('./02_noisy'):
        os.makedirs('./02_noisy')
    if not os.path.exists('./03_output'):
        os.makedirs('./03_output')

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set up model and criterion
    model = init_model(model_class, restore_path, restore_required=False, latent_dim=512)
    criterion = init_loss('vae', reconstruction_loss_type='mse')

    # load data
    data_loader, _, _, _, data_minval, data_maxval = \
        init_data_loader(dataset_key, batch_size, dataset_path)

    # training loop
    warning_displayed = False
    original, noisy, output = None, None, None
    model_optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        mean_loss, total_num_examples = 0, 0
        for data in data_loader:
            original = data.float()
            if not model.is_convolutional:
                original = original.view(original.size(0), -1)
            if torch.cuda.is_available():
                original = original.cuda()

            # apply noise
            if noise_type == 'mn':
                noisy, _ = zero_mask(original, zero_frac)
            elif noise_type == 'gs':
                noisy, _ = add_gaussian(original, gaussian_stdev)
            elif noise_type == 'sp':
                noisy, _ = salt_and_pepper(original, sp_frac, data_minval, data_maxval)
            else:
                if not warning_displayed:
                    print('unrecognized noise type: %r' % (noise_type,))
                    print('using clean image as input')
                    warning_displayed = True
                noisy = original
            if torch.cuda.is_available():
                noisy = noisy.cuda()

            # =============== forward ===============
            output, mean, log_var = model(noisy)
            loss = criterion(output, original, mean, log_var)
            batch_size_ = original.size(0)  # might be undersized last batch
            total_num_examples += batch_size_
            # assumes `loss` is sum for batch
            mean_loss += (loss - mean_loss * batch_size_) / total_num_examples

            # =============== backward ==============
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

        # =================== log ===================
        print('epoch {}/{}, loss={:.6f}'.format(epoch + 1, num_epochs, mean_loss.item()))
        if epoch % log_freq == 0 or epoch == num_epochs - 1:
            # save images
            to_save = [
                (to_img(original.data.cpu()), './01_original', 'original'),
                (to_img(noisy.data.cpu()), './02_noisy', 'noisy'),
                (to_img(output.data.cpu()), './03_output', 'output'),
            ]
            for img, folder, desc in to_save:
                save_image_wrapper(img, os.path.join(folder, '{}_{}.png'.format(desc, epoch + 1)))

            # save model(s)
            torch.save(model.state_dict(), save_path)
            print('[o] saved model to %s' % save_path)
