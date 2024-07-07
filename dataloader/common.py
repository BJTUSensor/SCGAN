import random
import numpy as np
import torch


def get_patch(img_in, img_tar, patch_size, isPair):
    if isPair:
        assert img_in.shape == img_tar.shape
        h, w = img_in.shape[:2]
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        img_in = img_in[y:y + patch_size, x:x + patch_size]
        img_tar = img_tar[y:y + patch_size, x:x + patch_size]
    else:
        h, w = img_in.shape[:2]
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        img_in = img_in[y:y + patch_size, x:x + patch_size]

        h, w = img_tar.shape[:2]
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        img_tar = img_tar[y:y + patch_size, x:x + patch_size]

    return img_in, img_tar


def add_noise(x, noise='.'):
    # add Gaussian white noise to x at certain sigma
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1])
        noises = np.random.normal(scale=noise_value, size=x.shape)
        noises = noises.round()
        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        # x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        # torch.from_numpy do not support negative strides
        if hflip: img = img[:, ::-1, :].copy()
        if vflip: img = img[::-1, :, :].copy()
        if rot90: img = img.transpose(1, 0, 2).copy()
        return img

    return [_augment(_l) for _l in l]


def normalization(data):
    MAX = np.max(data)
    MIN = np.min(data)
    data_norm = (data - MIN) / (MAX - MIN)
    return data_norm
