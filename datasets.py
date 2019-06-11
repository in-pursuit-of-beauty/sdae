import os
import h5py
import torch
import pickle
import imageio
import numpy as np
import scipy.io as sio
from torch.utils import data
from scipy.ndimage import rotate
from torchvision.datasets import MNIST
from skimage.util import view_as_windows


class OlshausenDataset(data.Dataset):
    """(Whitened) natural scene images.
    Available here: http://www.rctn.org/bruno/sparsenet.
    """

    def __init__(self, mat_path, patch_size=12, step_size=1, normalize=False):
        dataset = sio.loadmat(mat_path)
        images = np.ascontiguousarray(dataset['IMAGES'])  # shape: (512, 512, 10)
        self.patches = np.squeeze(view_as_windows(
            images, (patch_size, patch_size, 10), step=step_size))  # shape: (., ., PS, PS, 10)
        self.patches = self.patches.transpose((0, 1, 4, 2, 3))
        self.patches = self.patches.reshape((-1, patch_size, patch_size))
        if normalize:
            # normalize to range [0, 1]
            _min = self.patches.min()
            _max = self.patches.max()
            self.patches = (self.patches - _min) / (_max - _min)
        if self.patches.dtype != np.float:
            print('converting data type from %r to np.float' % self.patches.dtype)
            self.patches = self.patches.astype(np.float)
        print('image statistics:')
        print('min: %r, mean: %r, max: %r'
              % (self.patches.min(), self.patches.mean(), self.patches.max()))

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, index):
        return self.patches[index, :, :], index  # (image, "label")

    def get_minval(self):
        return self.patches.min()

    def get_maxval(self):
        return self.patches.max()


class MNISTVariant(MNIST):
    """Modified MNIST, or original MNIST if variant=None.
    Based on PyTorch's MNIST code at torchvision/datasets/mnist.py."""

    variant_options = (
        'rot',
        'bg_rand',
        'bg_rand_rot',
    )

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 variant=None,
                 generate=False):

        super(MNIST, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        # check for existence of original MNIST dataset
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        if variant:
            self.variant = variant.lower()
            if self.variant not in self.variant_options:
                if self.variant.startswith('mnist'):
                    self.variant = self.variant[5:]
                if self.variant.startswith('_'):
                    self.variant = self.variant[1:]
            assert self.variant in self.variant_options
            self.variant_folder = os.path.join(self.root, 'MNIST_%s' % self.variant, 'processed')

            data_path = os.path.join(self.variant_folder, data_file)
            if not os.path.exists(data_path) or generate:
                self.generate_variant_dataset(self.variant)
        else:
            self.variant, self.variant_folder = None, None
            data_path = os.path.join(self.processed_folder, data_file)

        self.data, self.targets = torch.load(data_path)

    def generate_variant_dataset(self, variant):
        """Generate a dataset corresponding to the given MNIST variant.

        The modified MNIST data will be saved in a similar fashion to
        that of the original MNIST dataset. Also, presumably some randomness will be
        involved, meaning the dataset will change every time this function is called.
        """
        # process and save as torch files
        print('Generating...')

        if not os.path.exists(self.variant_folder):
            os.makedirs(self.variant_folder)

        def _rot(image_data):
            """Destructive rotation."""
            for i in range(image_data.shape[0]):
                rand_deg = np.random.random() * 360.0
                image_data[i] = rotate(image_data[i], rand_deg, reshape=False)

        def _bg_rand(image_data):
            """Destructive random background."""
            noise = np.random.randint(
                0, 256, image_data.shape, dtype=image_data.dtype)
            image_data[image_data == 0] = noise[image_data == 0]

        for data_file in (self.training_file, self.test_file):
            # load original MNIST data
            data, targets = torch.load(os.path.join(self.processed_folder, data_file))

            modified_data = data.numpy()  # shape: (n, 28, 28)
            if variant == 'rot':
                _rot(modified_data)
            elif variant == 'bg_rand':
                _bg_rand(modified_data)
            elif variant == 'bg_rand_rot':
                _rot(modified_data)
                _bg_rand(modified_data)

            with open(os.path.join(self.variant_folder, data_file), 'wb') as f:
                torch.save((torch.from_numpy(modified_data), targets), f)

        print('Done!')
        print('Saved dataset to %s.' % self.variant_folder)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')


class CUB2011Dataset(data.Dataset):
    """Caltech-UCSD Birds-200-2011.
    Available here: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
    """
    training_file = 'train.h5'
    eval_file = 'eval.h5'

    def __init__(self, cub_folder, train=True, normalize=False):
        """CUB_FOLDER should contain the following items (among other things):
        images/, images.txt, train_test_split.txt, classes.txt, image_class_labels.txt."""
        super(CUB2011Dataset, self).__init__()
        self.cub_folder = cub_folder
        self.train = train  # training set or test set

        self.classes_path = os.path.join(self.cub_folder, 'classes.pkl')
        if not os.path.exists(self.classes_path):
            self.process_classes()
        with open(self.classes_path, 'rb') as f:
            self.classes = pickle.load(f)

        if self.train:
            data_path = os.path.join(self.cub_folder, self.training_file)
        else:
            data_path = os.path.join(self.cub_folder, self.eval_file)

        if not os.path.exists(data_path):
            self.process_images_and_labels()

        h5f = h5py.File(data_path, 'r')
        self.images = h5f['images'][:]  # shape: (n, h, w, 3)
        self.labels = h5f['labels'][:]  # shape: (n,)
        h5f.close()

        if normalize:
            # normalize to range [0, 1]
            _min = self.images.min()
            self.images = (self.images - _min) / (self.images.max() - _min)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]  # (image, label)

    def process_classes(self):
        classes = {}
        with open(os.path.join(self.cub_folder, 'classes.txt')) as f:
            for line in f:
                class_id, class_name = line.strip().split()
                classes[class_id - 1] = class_name
        with open(self.classes_path, 'wb') as f:
            pickle.dump(classes, f, pickle.HIGHEST_PROTOCOL)

    def process_images_and_labels(self):
        train_test_split = {}
        with open(os.path.join(self.cub_folder, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_training_image = line.strip().split()
                train_test_split[image_id - 1] = bool(is_training_image)

        image_class_labels = {}
        with open(os.path.join(self.cub_folder, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.strip().split()
                image_class_labels[image_id - 1] = class_id - 1

        train_images, train_labels = [], []
        eval_images, eval_labels = [], []
        with open(os.path.join(self.cub_folder, 'images.txt')) as f:
            for line in f:
                image_id, image_name = line.strip().split()
                image_path = os.path.join('images', image_name)
                if train_test_split[image_id - 1]:
                    train_images.append(imageio.imread(image_path))
                    train_labels.append(image_class_labels[image_id - 1])
                else:
                    eval_images.append(imageio.imread(image_path))
                    eval_labels.append(image_class_labels[image_id - 1])
        train_images, train_labels = [np.array(ta) for ta in (train_images, train_labels)]
        eval_images, eval_labels = [np.array(ea) for ea in (eval_images, eval_labels)]

        train_eval_triplets = [
            (os.path.join(self.cub_folder, self.training_file), train_images, train_labels),
            (os.path.join(self.cub_folder, self.eval_file), eval_images, eval_labels)
        ]
        for data_path, images, labels in train_eval_triplets:
            h5f = h5py.File(data_path, 'w')
            h5f.create_dataset('images', data=images)
            h5f.create_dataset('labels', data=labels)
            h5f.close()

    def get_minval(self):
        return self.images.min()

    def get_maxval(self):
        return self.images.max()
