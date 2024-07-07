import os
import scipy.io as io
import torch
import numpy as np
from dataloader import common
import torch.utils.data as data


class PairedFolders(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.name = 'Paired Folders for training'

        self._set_filesystem(opt.dir_data)
        self.n_paths, self.c_paths = self._scan()

        print('Original len {}, {} steps/epoch, batch size {}'.format( \
            len(self.n_paths), opt.steps_per_epoch, opt.batch_size))

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
        n = min(len(n_paths), len(c_paths))  #选取noise文件和clean文件中长度最小的
        return n_paths[0:n], c_paths[0:n]

    def __getitem__(self, idx):
        n_img, c_img, n_path, c_path = self._load_file(idx)
        # 获取patch
        # n_img, c_img = common.get_patch(n_img, c_img, self.opt.patch_size, isPair=True)
        # 数据增强
        # n_img, c_img = common.augment([n_img, c_img])
        # 最大最小归一化
        n_img, c_img = self.normalization(n_img, c_img)
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
        c_path = self.c_paths[idx]
        x = os.path.split(n_path)[-1]
        y = os.path.split(c_path)[-1]
        assert x.split('.')[-1] == y.split('.')[-1]
        n_img = io.loadmat(n_path)
        n_img = n_img['data']
        c_img = io.loadmat(c_path)
        c_img = c_img['data']
        return n_img, c_img, n_path, c_path

    def normalization(self, n_img, c_img):
        n_img = common.normalization(n_img)
        c_img = common.normalization(c_img)
        return n_img, c_img
