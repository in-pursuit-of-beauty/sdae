# SDAE / [ML4ART FORK]

### Setup

Install [PyTorch](https://pytorch.org/get-started/locally/), NumPy, and
[other common packages](https://github.com/ohjay/sdae/blob/master/requirements.txt) if you don't have them already.
```
pip install -r requirements.txt
```

Download the [RESISC45 dataset](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)
and place it in the `data` folder. The final path should be `sdae/data/NWPU-RESISC45`.

### Usage

Train the denoising variational autoencoder:
```
python3 train.py --batch_size 32 \
                 --learning_rate 0.001 \
                 --num_epochs 5000 \
                 --model_class CVAE \
                 --dataset_key resisc \
                 --noise_type gs \
                 --gaussian_stdev 0.4 \
                 --save_path ./ckpt/cvae.pth \
                 --weight_decay 0.0000001 \
                 --dataset_path data/NWPU-RESISC45
```

Generate samples using the trained model:
```
python3 generate_samples.py --model_class CVAE \
                            --restore_path ./ckpt/cvae.pth \
                            --num 30 \
                            --sample_h 256 \
                            --sample_w 256 \
                            --out_dir samples
```

### Associated Visuals

![ofmfts](https://user-images.githubusercontent.com/8358648/59959318-cd262800-9482-11e9-99e4-323066773608.png)
