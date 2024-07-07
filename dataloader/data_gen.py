import random

from utils.tools import *
# code: UTF-8
def dir_scan(img_input_dir):
    img_list = sorted(
        [os.path.join(img_input_dir, x)
         for x in os.listdir(img_input_dir)
         if (x.endswith('.mat') and not (x.startswith('._')))], reverse=True
    )
    return img_list

class SimuBGS(object):
    def __init__(self, save_dir, points=3500, num=1, snr=None):
        self.save_dir = save_dir
        check_dir_or_recreate(self.save_dir)
        self.snr = snr

        # self.gen_change_data(points, num)
        # self.gen_train_data(points, num)
        # self.gen_random_data(points, num)
        # self.gen_hot_data(points, num)
        self.simu_data(mode='random', points=points, num=num)

    def gen_change_data(self, points, num):
        freq_min = 10.501
        freq_max = 11.2 + 0.00001
        freq_step = 0.001
        freq = np.arange(freq_min, freq_max, freq_step)
        freq_bfs_min = 10.780
        freq_bfs_max = 11.0
        freq_bfs = np.arange(freq_bfs_min, freq_bfs_max, freq_step)
        sw_min = 0.07
        sw_max = 0.25
        sw_step = 0.002
        # sw = np.arange(sw_min, sw_max, sw_step)
        peak_gain = 1

        print("genetating BGS train data")
        cnt = 0
        for n in range(int(num)):
            sw = np.random.rand() * (sw_max - sw_min) + sw_min
            bfs = np.random.rand() * (freq_bfs_max - freq_bfs_min) + freq_bfs_min
            data = np.zeros((len(freq), points))
            noisy = np.zeros((len(freq), points))
            for i in range(points):
                BGS = self.lorentz_func(peak_gain, freq, bfs, sw)
                data[:, i] = BGS

                # if self.snr:
                #     no = self.add_awgn(BGS, self.snr)
                #     noisy[:, i] = no
            cnt += 1
            if cnt > 0:
                filename = '{:0>4d}'.format(cnt) + '.mat'
                scio.savemat(os.path.join(self.save_dir, filename), {'data': data})

                if self.snr:
                    scio.savemat(os.path.join(self.save_dir.replace('clean', 'noisy'), filename),
                                 {'data': noisy, 'bfs': bfs, 'sw': sw})
        print('done')
        print(cnt)

    def gen_train_data(self, points, num):
        freq_min = 10.751
        freq_max = 10.950 + 0.00001
        freq_step = 0.001
        freq = np.arange(freq_min, freq_max, freq_step)
        freq_bfs_min = 10.780
        freq_bfs_max = 10.920
        freq_bfs = np.arange(freq_bfs_min, freq_bfs_max, freq_step)
        sw_min = 0.03
        sw_max = 0.08
        sw_step = 0.002
        sw = np.arange(sw_min, sw_max, sw_step)
        peak_gain = 1

        print("genetating BGS train data")
        cnt = 0
        for n in range(int(num)):
            for v in sw:
                for bfs in freq_bfs:
                    data = np.zeros((len(freq), points))
                    noisy = np.zeros((len(freq), points))
                    for i in range(points):
                        BGS = self.lorentz_func(peak_gain, freq, bfs, v)
                        data[:, i] = BGS

                        if self.snr:
                            no = self.add_awgn(BGS, self.snr)
                            noisy[:, i] = no
                    cnt += 1
                    if cnt > 300:
                        filename = '{:0>4d}'.format(cnt) + '.mat'
                        scio.savemat(os.path.join(self.save_dir, filename), {'data': data})

                        if self.snr:
                            scio.savemat(
                                os.path.join(self.save_dir.replace('clean', 'noisy' + str(self.snr)), filename),
                                {'data': noisy})
        print('done')
        print(cnt)

    def gen_random_data(self, points, num):
        freq_min = 10.501
        freq_max = 11.2 + 0.00001
        freq_step = 0.001
        freq = np.arange(freq_min, freq_max, freq_step)
        freq_bfs_min = 10.780
        freq_bfs_max = 10.920
        sw_min = 0.03
        sw_max = 0.08
        peak_gain = 1
        snr_min = -10
        snr_max = 40

        print("genetating BGS train random data")
        cnt = 600
        for n in range(int(num)):
            data = np.zeros((len(freq), points))
            noisy = np.zeros((len(freq), points))
            bfs = np.random.rand(points) * (freq_bfs_max - freq_bfs_min) + freq_bfs_min
            sw = np.random.rand(points) * (sw_max - sw_min) + sw_min
            # snr_dB = np.random.rand() * (snr_max - snr_min) + snr_min
            snr_dB = 15
            for i in range(points):
                BGS = self.lorentz_func(peak_gain, freq, bfs[i], sw[i])
                data[:, i] = BGS

                if self.snr:
                    no = self.add_awgn(BGS, snr_dB)
                    noisy[:, i] = no
            cnt += 1
            if cnt > 0:
                filename = '{:0>4d}'.format(cnt) + '.mat'
                scio.savemat(os.path.join(self.save_dir, filename), {'data': data, 'bfs': bfs, 'sw': sw})

                if self.snr:
                    scio.savemat(os.path.join(self.save_dir.replace('clean', 'noisy'), filename), {'data': noisy})

        print("done")
        print(cnt)

    def gen_hot_data(self, points, num):
        fiber_length_range = [40, 60]
        freq_min = 10.751
        freq_max = 10.950 + 0.00001
        freq_step = 0.001
        freq = np.arange(freq_min, freq_max, freq_step)
        freq_bfs_min = 10.780
        freq_bfs_max = 10.920
        sw_min = 0.03
        sw_max = 0.08
        peak_gain = 1
        snr_min = 10
        snr_max = 25
        print("genetating BGS train hot data")
        cnt = 1200
        for n in range(int(num)):
            data = np.zeros((len(freq), points))
            noisy = np.zeros((len(freq), points))
            fiber_len_list = np.random.randint(low=fiber_length_range[0], high=fiber_length_range[1] + 1, size=10)
            bfs_list = np.random.rand(10) * (freq_bfs_max - freq_bfs_min) + freq_bfs_min
            sw_list = np.random.rand(10) * (sw_max - sw_min) + sw_min
            snr_list = np.random.rand(10) * (snr_max - snr_min) + snr_min

            bgs_num = 0
            small_piece_num = 0
            while bgs_num < points:
                if (bgs_num + fiber_len_list[small_piece_num]) > points:
                    length = points - bgs_num
                else:
                    length = fiber_len_list[small_piece_num]
                bfs = bfs_list[small_piece_num]
                sw = sw_list[small_piece_num]
                # snr = snr_list[small_piece_num]
                snr = 15
                for small_n in range(length):
                    BGS = self.lorentz_func(peak_gain, freq, bfs, sw)
                    data[:, bgs_num] = BGS

                    if self.snr:
                        no = self.add_awgn(BGS, snr)
                        noisy[:, bgs_num] = no

                    bgs_num += 1
                small_piece_num += 1

            cnt += 1
            if cnt > 0:
                filename = '{:0>4d}'.format(cnt) + '.mat'
                scio.savemat(os.path.join(self.save_dir, filename), {'data': data, 'bfs': bfs, 'sw': sw})

                if self.snr:
                    scio.savemat(os.path.join(self.save_dir.replace('clean', 'noisy'), filename), {'data': noisy})

        print("done")
        print(cnt)

    def simu_data(self, mode, points, num):
        if mode == 'random':
            # 扫频范围 [10550, 10850]MHz step:2 ~151
            freq_min = 10.751
            freq_max = 10.950 + 0.000001
            freq_step = 0.001
            freq = torch.arange(freq_min, freq_max, freq_step)
            # BFS范围 [10570, 10830]MHz
            freq_bfs_min = 10.85
            freq_bfs_max = 10.92
            # FWHM范围 [20, 80]MHz
            sw_min = 0.03
            sw_max = 0.095
            snr_min = 1
            snr_max = 25
            # 模拟数据
            # self.logger.info("generating BGS data")
            for n in range(int(num)):
                BGSs = np.zeros((len(freq), points))
                noisy = np.zeros((len(freq), points))
                peak_intensity = np.random.rand(points) + 0.00001
                freq_bfs = np.random.rand(points) * (freq_bfs_max - freq_bfs_min) + freq_bfs_min
                snr = 15
                sw = np.random.rand(points) * (sw_max - sw_min) + sw_min
                for i in range(points):
                    BGS = np.array(self.lorentz_func(peak_intensity[i], freq, freq_bfs[i], sw[i]))
                    BGSs[:, i] = BGS

                    if self.snr:
                         no = self.add_awgn(BGS, snr)
                         noisy[:, i] = no
                # 保存模拟数据
                filename = str(n) + '.mat'
                scio.savemat(os.path.join(self.save_dir, filename), {'data': BGSs})
                scio.savemat(os.path.join(self.save_dir.replace('clean', 'noisy'), filename), {'data': noisy})

        if mode == 'ascend':
            # 扫频范围 [10550, 10850]MHz step:2 ~151
            freq_min = 10.550
            freq_max = 10.850
            freq_step = 0.002
            freq = torch.arange(freq_min, freq_max, freq_step)
            # BFS范围 [10570, 10820]MHz  step:0.01
            freq_bfs_min = 10.570
            freq_bfs_max = 10.830
            freq_bfs_step = (freq_bfs_max - freq_bfs_min) / points
            freq_bfs = torch.arange(freq_bfs_min, freq_bfs_max + freq_bfs_step, freq_bfs_step)
            # FWHM范围 [20, 80]MHz
            sw_min = 0.03
            sw_max = 0.10

            # gB峰值范围 [0, 2]
            peak_intensity = 1
            # 模拟数据
            self.logger.info("genetating BGS data")
            for n in range(num):
                BGSs = np.zeros((len(freq), points))
                for i in range(points):
                    sw = 0.055
                    BGS = self.lorentz_func(peak_intensity, freq, freq_bfs[i], sw)
                    BGSs[:, i] = BGS
                # 保存模拟数据
                filename_clean = str(n) + '.mat'
                scio.savemat(os.path.join(self.save_dir, filename_clean), {'data': BGSs})
                self.logger.info('generate num {}/{} end.'.format(n + 1, num))

        if mode == 'hotspot':
            fiber_length_range = [300, 500]
            expected_sr = 1
            sweep_freq_range = [10.751, 10.950]  # GHz
            sweep_freq_step = 0.001  # GHz
            sweep_freq_size = 201
            bfs_range = [10.80, 10.90]  # GHz
            sw_range = [0.025, 0.035]  # MHz
            # gain_intensity_range = [0.7, 2]
            gain_intensity = 1

            bgs_x = np.linspace(sweep_freq_range[0], sweep_freq_range[1], sweep_freq_size)
            # 模拟数据
            self.logger.info("genetating BGS data")
            for n in range(int(num)):
                BGSs = np.zeros((len(bgs_x), points))
                fiber_length_arr = np.random.randint(low=fiber_length_range[0], high=fiber_length_range[1] + 1,
                                                     size=225)
                bfs_arr = np.random.rand(255) * (bfs_range[1] - bfs_range[0]) + bfs_range[0]
                # sw_arr = np.random.rand(255) * (sw_range[2] - sw_range[0]) + sw_range[0]
                # gain_intensity_arr = np.random.rand(points) * (gain_intensity_range[2] - gain_intensity_range[0]) + gain_intensity_range[0]

                bgs_num = 0
                small_piece_num = 0
                while bgs_num < points:
                    if (bgs_num + fiber_length_arr[small_piece_num]) > points:
                        length = points - bgs_num
                    else:
                        length = fiber_length_arr[small_piece_num]
                    bfs = bfs_arr[small_piece_num]
                    # sw = sw_arr[small_piece_num]S
                    sw = 0.055
                    for small_n in range(length):
                        # bgs = self.lorentz_func(gain_intensity_arr[bgs_num], bgs_x, bfs, sw)
                        bgs = self.lorentz_func(gain_intensity, bgs_x, bfs, sw)
                        BGSs[:, bgs_num] = bgs
                        bgs_num += 1
                    small_piece_num += 1
                # 对BGSs预处理（加噪和归一化）
                BGSs_clean_norm, BGSs_noise_norm = self.proprocessing(BGSs)
                # 保存模拟数据
                filename_clean = str(n) + '.mat'
                scio.savemat(os.path.join(self.save_dir, filename_clean),
                             {'data_clean': BGSs_clean_norm, 'data_noise': BGSs_noise_norm})
                self.logger.info('generate num {}/{} end.'.format(n + 1, num))

    def lorentz_func(self, peak_gain, freq, freq_bfs, sw):
        BGS = peak_gain / (1 + ((freq - freq_bfs) / (sw / 2)) ** 2)
        return BGS

    # 给每条BGS加噪声
    def add_awgn(self, data, snr):
        snr = 10 ** (snr / 10.0)
        _signal = np.var(data)
        _noise = _signal / snr
        noise = np.random.randn(len(data)).astype(np.float32) * np.sqrt(_noise)  # 返回固定维度具有标准正态分布的随机值
        return data + noise


import matplotlib.pyplot as plt
import scipy.io as io
if __name__ == '__main__':
    input_dir = '../data/real/train/test/clean'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        os.makedirs(input_dir.replace('clean', 'noisy'))
    #

    # img_n = dir_scan(noisy_dir)
    # snr_min = 1
    # snr_max = 25
    # for i in tqdm(range(len(img_list))):
    #     img = io.loadmat(img_list[i])
    #     img_c = img['data']
    #     noisy = np.zeros((200, 200))
    #     snr = np.random.rand() * (snr_max - snr_min) + snr_min
    #     for j in range(200):
    #         no = add_awgn(img_c[:,j], snr)
    #         noisy[:, j] = no
    #     filename = img_list[i].split('/')[-1].split('\\')[-1]
    #     scio.savemat(os.path.join(save_dir, filename), {'data': noisy})
    # print(random.randint(5,15))
    SimuBGS(input_dir, points=200, num=1, snr=True)
    # name = 'train'
    # file_dir = 'data/3-hotspot/{}/clean'.format(name)
    # list = get_file_list(file_dir, all_data=False, suffix=['mat'])
    # list.sort()
    # f = open('data/3-hotspot/{}.txt'.format(name), 'w')
    # f.write('\n'.join(list))
    # f.close()
    # path = '../data/real/test/noisy_raw/0029.mat'
    # img_n_dn = scio.loadmat(path)
    # data = img_n_dn['data'][30:670, :]
    # freq_min = 10.531
    # freq_max = 11.17 + 0.00001
    # freq_step = 0.001
    # freq = np.arange(freq_min, freq_max, freq_step)
    # dn_bfs, dn_snr = get_bfs_snr(freq, data)
    # x = np.arange(data.shape[1])
    # plt.plot(x, dn_snr)
    # plt.show()
    # print(torch.rand(1))
