import os
import shutil
import scipy.io as scio
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import leastsq


# import matplotlib.pyplot as plt


def check_dir_or_recreate(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)


def get_file_list(file_dir, all_data=False, suffix=['mat']):
    if not os.path.exists(file_dir):
        print('path {} is not exist'.format(file_dir))
        return []
    img_list = []

    for root, sdirs, files in os.walk(file_dir):
        if not files:
            continue
        for filename in files:
            filepath = root + '/' + filename
            if all_data or filename.split('.')[-1] in suffix:
                img_list.append(filepath)
    return img_list


def get_data(data_path, norm=0):
    suffix = data_path.split('/')[-1].split('.')[-1]
    if suffix == 'mat':
        data_mat = scio.loadmat(data_path)
        data = data_mat['data']

        data = torch.Tensor(data.astype(np.float32)).unsqueeze(0)

    if norm:
        min_v = data.min()
        max_v = data.max()
        data = (data - min_v) / (max_v - min_v) * norm

    return data


def save_data(save_dir, data_name, data):
    save_path = os.path.join(save_dir, data_name + '.mat')
    scio.savemat(save_path, {'data': data})


def norm_data(data):
    min_v = data.min()
    max_v = data.max()
    data = (data - min_v) / (max_v - min_v)
    return data


def lorentz_fit(x, y):
    p3 = ((np.max(x) - np.min(x)) / 10) ** 2
    p2 = (np.max(x) + np.min(x)) / 2
    p1 = np.max(y) * p3
    c = np.min(y)

    p0 = np.array([p1, p2, p3, c], dtype=np.float64)

    args, ier = leastsq(func=error_func, x0=p0, args=(x, y), maxfev=200000)

    '''
    p0 = np.array([p1, p2, p3], dtype=np.float64)
    args, _ = curve_fit(f=lorentz_func2, xdata=x, ydata=y, p0=p0, bounds=([-np.inf, 10, -np.inf], [np.inf, 11, np.inf]))
    '''
    return lorentz_func(args, x), args


def error_func(p, x, z):
    return z - lorentz_func(p, x)


def lorentz_func(p, x):
    return p[0] / ((x - p[1]) ** 2 + p[2])


def get_snr(y, y_fit, args):
    y_fit_max = lorentz_func(args, args[1])
    variance = np.var(y - y_fit)
    snr = y_fit_max ** 2 / variance
    snr_dB = 10 * np.log10(snr)
    return snr_dB


def get_bfs_snr(freqs, data):
    bfs = np.zeros(data.shape[1])
    snr = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        y_fit, args = lorentz_fit(freqs, data[:, i])
        bfs[i] = args[1]
        snr[i] = get_snr(data[:, i], y_fit, args)
    return bfs, snr


def get_rmse_sd(label, output):
    rmse = RMSE(label, output)
    sd = SD(output)
    return rmse, sd


# y是真实值，y_fit是测量值，计算结果是测量值与真实值的误差 (均方根误差)
def RMSE(y, y_fit):
    rmse = np.sqrt(np.mean((y - y_fit) ** 2))
    return rmse


# 描述数据的波动程度 (标准差)
def SD(y):
    sd = np.sqrt(np.var(y))
    return sd


# def plot_metric(save_path, name, model, x, noisy, denoise):
#     plt.figure()
#     plt.xlabel('points')
#     plt.ylabel(name)
#     plt.title(name)
#     plt.plot(x, denoise, label=model, color='red')
#     plt.plot(x, noisy, label='noisy', color='black')
#     plt.legend()
#     # plt.show()
#     f = plt.gcf()
#
#     f.savefig(save_path)
#     f.clear()


def get_patch(data, patch_size):
    h, w = data.shape[:2]
    x = np.random.randint(0, w - patch_size)
    y = np.random.randint(0, h - patch_size)
    data_patch = data[y:y + patch_size, x:x + patch_size]  #截取中心频率所在位置
    return data_patch

def add_awgn(data, snr):
    snr = 10 ** (snr / 10.0)
    _signal = np.var(data)
    _noise = _signal / snr
    noise = np.random.randn(len(data)).astype(np.float32) * np.sqrt(_noise)  # 返回固定维度具有标准正态分布的随机值
    return data + noise

# import h5py
# 处理HongKonng wangyuyao数据
def process_real_data(data_path, save_path):
    cnt = 0
    patch_size = 128
    # snr_min = 1
    # snr_max = 25
    file_list = get_file_list(data_path)
    for inp_path in tqdm(file_list):
        data_mat = scio.loadmat(inp_path)
        # data_mat = h5py.File(inp_path)
        # data = np.transpose(data_mat['data1'])
        data = data_mat['data']
        # data_norm = np.zeros(data.shape)
        # for i in range(700):
        #      data_norm[i, :] = data[i, :] / np.mean(data[i, :]) - 1
        # data = data_norm[:, 10000:13200]
        for i in range(15):
            patch = get_patch(data, patch_size=patch_size)
            cnt += 1
            scio.savemat(os.path.join(save_path, '{:0>4d}.mat'.format(cnt)), {'data': patch})
        # freq_min = 10.501
        # freq_max = 11.2 + 0.00001
        # freq_step = 0.001
        # freq = np.arange(freq_min, freq_max, freq_step)
        # dn_bfs, dn_snr = get_bfs_snr(freq, data)

        # noisy = np.zeros((len(data), len(data)))
        # snr = np.random.rand() * (snr_max - snr_min) + snr_min
        # for i in range(len(data)):
        #     no = add_awgn(data[:,i], snr)
        #     noisy[:, i] = no
        # scio.savemat(os.path.join(save_path, '{}.mat'.format(inp_path.split('/')[-1].split('.')[-2])), {'data': data, 'bfs':dn_bfs,'snr':dn_snr})

from preprocess.PM import *
from preprocess.NoiseEstimate import *
if __name__ == '__main__':
    # data_path = 'D:\\lk\\SCGAN\\data\\real\\test\\30'
    # save_path = 'D:\\lk\\SCGAN\\data\\real\\test\\30_new\\温度_30_average_time_500_1.mat'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # process_real_data(data_path, save_path)
    # BGS = scio.loadmat(save_path)['data']
    # sigma = NoiseLevel(BGS)
    # Thres = 1000 * sigma
    #
    # F = anisodiff2D
    # BGS1 = F(BGS, num_iter=7, kappa=Thres, delta_t=1 / 7, option=1)
    # freq_min = 10.501
    # freq_max = 11.2 + 0.00001
    # freq_step = 0.001
    # freq = np.arange(freq_min, freq_max, freq_step)
    # dn_bfs, dn_snr = get_bfs_snr(freq, BGS1)
    #
    # scio.savemat(os.path.join(save_path),
    #              {'data_raw': BGS, 'data_de':BGS1,'snr': dn_snr, 'bfs': dn_bfs})

    data_path = 'D:\\lk\\SCGAN\\data\\real\\train\\raw'
    save_path = 'D:\\lk\\SCGAN\\data\\real\\train\\small'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    process_real_data(data_path,save_path)