import os
import torch
import imageio
import numpy as np
from torch.utils import data


class RESISC45Dataset(data.Dataset):
    """RESISC45.
    Available here: http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html.
    """

    def __init__(self, resisc_folder, normalize=False):
        """RESISC_FOLDER should contain a subfolder for each class."""
        super(RESISC45Dataset, self).__init__()
        self.resisc_folder = resisc_folder

        paths = []
        for image_type in os.listdir(resisc_folder):
            if image_type == 'medium_residential':
                # only use "medium residential" for now
                image_type_folder = os.path.join(resisc_folder, image_type)
                for fname in os.listdir(image_type_folder):
                    paths.append(os.path.join(image_type_folder, fname))
        print('Registered %s images in dataset.' % len(paths))

        print('==> Reading images...')
        self.images = []
        for path in paths:
            image = imageio.imread(path, as_gray=True)
            image = (image - image.min()) / (image.max() - image.min())
            image = np.expand_dims(image, 0)  # reshape to (c, h, w)
            self.images.append(image)
        self.images = np.array(self.images)
        print('Done reading images.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

    def get_minval(self):
        return self.images.min()

    def get_maxval(self):
        return self.images.max()

    def get_data_dims(self):
        return 1, 256, 256  # (c, h, w)
