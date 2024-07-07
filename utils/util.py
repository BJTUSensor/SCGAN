from __future__ import print_function
import torch
import numpy as np
import os
import time


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def tensor2im(input_image):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    # img_data = image_tensor.squeeze(0).squeeze(0).cpu().numpy()
    # Tensor with shape (batch, channel, h, w)
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    if image_numpy.shape[2] == 1:
        h = image_numpy.shape[0]
        w = image_numpy.shape[1]
        image_numpy = image_numpy.reshape(h, w)
    return image_numpy


# def save_image(image_numpy, image_path):
#     h = image_numpy.shape[0]; w = image_numpy.shape[1]
#     io.imsave(image_path, image_numpy.reshape(h,w))
#     # image_pil = Image.fromarray(image_numpy)
#     # image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out





