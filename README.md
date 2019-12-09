# SDAE: ML4ART FORK

### Setup

Install [PyTorch](https://pytorch.org/get-started/locally/), NumPy, and
[other common packages](https://github.com/ohjay/sdae/blob/master/requirements.txt) if you don't have them already.
```
pip install -r requirements.txt
```

There is one dataset you'll need to download manually (see below).
I suggest you create a `data` folder and unpack the relevant files into it.
Later, you will be able to specify the dataset path as a command line argument.

<table>
  <tr>
    <td>RESISC45</td>
    <td><a href="http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html">Link</a></td>
  </tr>
</table>

### Usage

Train the denoising variational autoencoder:
```
python3 sdae.py \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --model_class MNISTSVAE \
    --dataset_key resisc \
    --noise_type gs \
    --gaussian_stdev 0.4 \
    --save_path ./ckpt/sdvae.pth \
    --weight_decay 0.0000001 \
    --vae_reconstruction_loss_type bce
```

### Associated Visuals

![ofmfts](https://user-images.githubusercontent.com/8358648/59959318-cd262800-9482-11e9-99e4-323066773608.png)
