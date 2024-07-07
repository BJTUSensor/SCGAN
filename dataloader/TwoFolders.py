import os
import scipy.io as io
from dataloader import common
import random
import torch
import torch.utils.data as data
import numpy as np

# Dataset是PyTorch中用于表示数据集的抽象类，它相当于一种列表结构，可以用来存储和访问数据样本。而Dataloader是一个用于加载数据的实用程序类，
# 它可以从Dataset中按照指定的批次大小和顺序加载数据。Dataloader还提供了多线程和多进程的功能，以加快数据加载速度。
# 因此，Dataset和Dataloader的区别在于，Dataset是用来存储和管理数据集的类，而Dataloader是用来加载数据的实用程序类。
class TwoFolders(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.name = 'Two Folders for training'

        self._set_filesystem(opt.dir_data)
        self.n_paths, self.c_paths = self._scan()

        print('Original len {}, {} steps/epoch, batch size {}'.format(len(self.n_paths), opt.steps_per_epoch,
                                                                      opt.batch_size))

    def _set_filesystem(self, dir_data):
        self.root = dir_data
        self.dir_n = os.path.join(self.root, self.opt.dataset_noisy)
        self.dir_c = os.path.join(self.root, self.opt.dataset_clean)
        print('==> Dataset: dir_n, dir_c')
        print(self.dir_n)
        print(self.dir_c)

    def _scan(self):
        n_paths = sorted(
            [os.path.join(self.dir_n, x) for x in os.listdir(self.dir_n)]
        )
        c_paths = sorted(
            [os.path.join(self.dir_c, x) for x in os.listdir(self.dir_c)]
        )
        n = min(len(n_paths), len(c_paths))
        return n_paths[0:n], c_paths[0:n]

    def __getitem__(self, idx):
        n_img, c_img, n_path, c_path = self._load_file(idx)
        # 获取patch
        # n_img, c_img = common.get_patch(n_img, c_img, self.opt.patch_size, isPair=False)
        # 数据增强
        # n_img, c_img = common.augment([n_img, c_img])
        # 最大最小归一化
        # n_img, c_img = self.normalization(n_img, c_img)
        # 转成 tensor
        n_img_tensor = torch.Tensor(n_img.astype(np.float32)).unsqueeze(0)
        c_img_tensor = torch.Tensor(c_img.astype(np.float32)).unsqueeze(0)
        return {'A': n_img_tensor, 'B': c_img_tensor, 'A_paths': n_path, 'B_paths': c_path}

    def __len__(self):
        return len(self.n_paths)

    def _get_index(self, idx):
        return idx % len(self.n_paths)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        n_path = self.n_paths[idx]
        c_path = self.c_paths[random.randint(0, len(self.n_paths) - 1)]
        n_img = io.loadmat(n_path)
        n_img = n_img['data']
        c_img = io.loadmat(c_path)
        c_img = c_img['data']
        return n_img, c_img, n_path, c_path

    def normalization(self, n_img, c_img):
        n_img = common.normalization(n_img)
        c_img = common.normalization(c_img)
        return n_img, c_img
