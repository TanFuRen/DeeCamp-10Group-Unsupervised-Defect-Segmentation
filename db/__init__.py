import cv2
import numpy as np
import torch
from .mvtec import MVTEC
from .mvtec import Preproc as MVTEC_pre


def training_collate(batch):
    """Custom collate fn for dealing with batches of images.

    Arguments:
        batch: (tuple) A tuple of tensor images

    Return:
        (tensor) batch of images stacked on their 0 dim
    """
    # imgs = list()
    # for img in batch:
    #     _c, _h, _w = img.shape
    #     imgs.append(img.view(1, _c, _h, _w))

    return torch.stack(batch, 0)


class Transform(object):
    def __init__(self, resize):
        self.resize = resize

    def __call__(self, image):
        image = cv2.resize(image, self.resize)
        image = image.astype(np.float32) / 255.
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        return image.unsqueeze(0)

