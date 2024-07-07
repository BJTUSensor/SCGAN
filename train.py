import random
import torch.backends.cudnn
from options import opt
from models import create_model
from dataloader import CreateDataLoader
from utils.visualizer import Visualizer
from utils.util import *
from utils.tools import *
import cv2

# setting random seed
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
torch.backends.cudnn.deterministic = True

def dir_scan(img_input_dir):
    img_list = sorted(
        [os.path.join(img_input_dir, x)
         for x in os.listdir(img_input_dir)
         if (x.endswith('.mat') and not (x.startswith('._')))], reverse=True
    )
    return img_list

def get_patch(full_input_img, patch_size):
    patch_input_imgs = []
    h, w = full_input_img.shape
    top = 0
    while top < h:
        left = 0
        new_h, new_w = patch_size, patch_size
        while left < w:
            if(left + new_w > w - 1):
                new_w = w - left
            if(top + new_h > h - 1):
                new_h = h - top
            patch_input_img = full_input_img[top:top + new_h, left:left + new_w]
            patch_input_imgs.append(patch_input_img)
            left += new_w
        top += new_h
    return np.array(patch_input_imgs, dtype=object)

def get_all(patch_input_imgs):
    for j in range(3):
        for i in range(1,11) :
            patch_input_imgs[j * 11] = np.concatenate((patch_input_imgs[j * 11],patch_input_imgs[i+j * 11]), 1)

    patch_input_img = np.concatenate((patch_input_imgs[0], patch_input_imgs[11]),0)
    patch_input_img = np.concatenate((patch_input_img, patch_input_imgs[22]),0)
    return np.array(patch_input_img)

def performance_evaluator(model, dir_noisy, save_results=True, dir_saving='.'):
    print('==> Dataset: dir_n, dir_c')
    print(dir_noisy)

    img_n_list = dir_scan(dir_noisy)

    snr = []
    ssim = []
    mse = []

    for i in tqdm(range(len(img_n_list))):
        img_n = scio.loadmat(img_n_list[i])['data']
        img_n_dn = model(img_n)

        # calculate psnr and save results
        freq_min = 10.701
        freq_max = 11.0 + 0.000001
        freq_step = 0.001
        freq = np.arange(freq_min, freq_max, freq_step)
        dn_bfs, dn_snr = get_bfs_snr(freq, img_n_dn)
        snr.append(dn_snr.mean())

        scio.savemat(os.path.join(dir_saving, os.path.split(img_n_list[i])[-1]),
                     {'data': img_n_dn, 'snr': dn_snr, 'bfs': dn_bfs})
        min_v = img_n_dn.min()
        max_v = img_n_dn.max()
        img_n_dn = (img_n_dn - min_v) / (max_v - min_v) * 255
        cv2.imwrite(os.path.join(dir_saving, os.path.split(img_n_list[i])[-1]).replace('.mat', '.png'), img_n_dn)

    avg_ssim = np.mean(ssim) if len(ssim) >= 1 else 'N.A.'
    avg_mse = np.mean(mse) if len(mse) >= 1 else 'N.A.'
    print('Results: avg_ssim:[{}], avg_mse:[{}], saving_results:[{}]'.format(avg_ssim, avg_mse, save_results))


def performance_evaluator2(model, dir_noisy, save_results=True, dir_saving='.'):
    print('==> Dataset: dir_n')
    print(dir_noisy)
    img_n_list = dir_scan(dir_noisy)

    snr = []
    for i in tqdm(range(len(img_n_list))):
        img_n = scio.loadmat(img_n_list[i])['data']
        img_n_dn=model(img_n)
        # img_n_patch = get_patch(img_n,300)
        # img_n_dns = []
        # for each in img_n_patch:
        #     img_n_dns.append(model(each))
        # img_n_dn = get_all(img_n_dns)
        # calculate psnr and save results
        freq_min = 10.501
        freq_max = 11.2 + 0.00001
        freq_step = 0.001
        freq = np.arange(freq_min, freq_max, freq_step)
        dn_bfs, dn_snr = get_bfs_snr(freq, img_n_dn)
        snr.append(dn_snr.mean())

        scio.savemat(os.path.join(dir_saving, os.path.split(img_n_list[i])[-1]),
                     {'data': img_n_dn, 'snr': dn_snr, 'bfs': dn_bfs})
        # min_v = img_n_dn.min()
        # max_v = img_n_dn.max()
        # img_n_dn = (img_n_dn - min_v) / (max_v - min_v) * 255
        # cv2.imwrite(os.path.join(dir_saving, os.path.split(img_n_list[i])[-1]).replace('.mat', '.png'), img_n_dn)



if __name__ == '__main__':
    torch.cuda.set_device(0)

    if opt.mode == 'NoiseModeling':
        if opt.isTrain:
            # Preparation
            train_loader = CreateDataLoader(opt).dataloader
            visualizer = Visualizer(opt)
            model = create_model(opt)
            model.setup(log_file=visualizer.log_name)

            for epoch in range(opt.which_epoch + 1, opt.epochs + 1):
                epoch_start_time = time.time()
                timer_data = timer()
                timer_model = timer()
                # load data
                for curr_step, data in enumerate(train_loader):
                    timer_data.hold()
                    model.set_input(data)
                    model.optimize_parameters(curr_step=curr_step, curr_epoch=epoch)
                    timer_model.hold()
                    # logging
                    if (curr_step + 1) % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        t_data = timer_data.release()
                        t_model = timer_model.release()
                        visualizer.print_current_losses(epoch, curr_step + 1, opt.steps_per_epoch, losses,
                                                        (t_data + t_model), t_data, t_model)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, (curr_step + 1) / len(train_loader), losses)
                    timer_data.tic()

                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result=True)
                model.save_networks(epoch)
                message = 'Model saved @ the end of epoch [%d/%d]. \t Time Taken: %d sec' % (
                epoch, opt.epochs, time.time() - epoch_start_time)
                print(message)
                with open(visualizer.log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
                ## update lr
                model.update_learning_rate(visualizer)
        else:
            # Prepare data
            print('==> Dataset: dir_n, dir_c')
            print('aaaaaaaaaaaaaaaaa')
            print(opt.dir_noisy)
            print(opt.dir_clean)

            img_c_list = dir_scan(opt.dir_clean)
            img_n_list = dir_scan(opt.dir_noisy)
            N_imgs = len(img_c_list)
            img_n_list = (img_n_list * (N_imgs // len(img_n_list) + 1))[0:N_imgs]

            print(os.path.split(opt.dir_clean)[0])
            print(os.path.split(opt.pre_train)[-1])
            dir_noise = os.path.join(os.path.split(opt.dir_clean)[0], 'experiments_cC', os.path.split(opt.pre_train)[-1], 'noise')
            dir_out_n = os.path.join(os.path.split(opt.dir_clean)[0], 'experiments_cC', os.path.split(opt.pre_train)[-1],'rec')
            print("dir_out_n:", dir_out_n)
            print("dir_out_n:", dir_noise)
            if not os.path.exists(dir_out_n):
                os.makedirs(dir_out_n)
            if not os.path.exists(dir_noise):
                os.makedirs(dir_noise)
            # Create model
            model = create_model(opt)
            model.setup()
            for i in tqdm(range(N_imgs)):
                # load clean
                img_c = scio.loadmat(img_c_list[i])['data']
                img_c = torch.Tensor(img_c.astype(np.float32)).unsqueeze(0)
                img_c = torch.reshape(img_c, (-1, img_c.shape[0], img_c.shape[1], img_c.shape[2]))

                # load noise images and set the size same as clean images
                img_n = scio.loadmat(img_n_list[i])['data']
                img_n = torch.Tensor(img_n.astype(np.float32)).unsqueeze(0)
                img_n = torch.reshape(img_n, (-1, img_n.shape[0], img_n.shape[1], img_n.shape[2]))

                model.set_input({'A': img_n, 'B': img_c})
                model.testing()
                img_c_n = tensor2im(model.c_B_n, opt)
                img_noise = tensor2im(model.n_A_map, opt)
                scio.savemat(os.path.join(dir_out_n, os.path.split(img_c_list[i])[-1]), {'data': img_c_n})
                scio.savemat(os.path.join(dir_noise, os.path.split(img_c_list[i])[-1]), {'data': img_noise})
    else:
        assert opt.mode == 'Denoising'
        if opt.isTrain:
            # Preparation
            train_loader = CreateDataLoader(opt).dataloader
            visualizer = Visualizer(opt)
            model = create_model(opt)
            model.setup(log_file=visualizer.log_name)
            # Training Procedure, 100 epochs, 4e3 steps/epoch, decay/10 epochs
            for epoch in range(1, opt.epochs + 1):
                # set timer
                epoch_start_time = time.time()
                timer_data = timer()
                timer_model = timer()

                # load data
                for curr_step, data in enumerate(train_loader):
                    timer_data.hold()
                    visualizer.reset()
                    # # forward and backward
                    timer_model.tic()
                    model.set_input(data)

                    model.optimize_parameters(curr_step=curr_step, curr_epoch=epoch)
                    timer_model.hold()
                    # logging
                    if (curr_step + 1) % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        t_data = timer_data.release()
                        t_model = timer_model.release()
                        visualizer.print_current_losses(epoch, curr_step + 1, opt.steps_per_epoch, losses,
                                                        (t_data + t_model), t_data, t_model)
                        if opt.display_id > 0:
                            # print('++++++++++++{}'.format(curr_step))
                            visualizer.plot_current_losses(epoch, (curr_step + 1) / len(train_loader), losses)
                    timer_data.tic()

                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result=True)
                model.save_networks(epoch)
                message = 'Model saved @ the end of epoch [%d/%d]. \t Time Taken: %d sec' % (
                epoch, opt.epochs, time.time() - epoch_start_time)
                print(message)
                with open(visualizer.log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
                # ## update lr
                model.update_learning_rate(visualizer)
        else:
            if opt.dn_methods == 'SCGAN':
                dir_out_n = os.path.join(os.path.split(opt.dir_noisy)[0], 'experiments', 'dn_results',
                                         os.path.split(opt.pre_train)[-1])
            else:
                dir_out_n = os.path.join(os.path.split(opt.dir_noisy)[0], 'experiments', 'dn_results',
                                         os.path.split(opt.dir_noisy)[-1] + '_' + opt.dn_methods)

            if not os.path.exists(dir_out_n):
                os.makedirs(dir_out_n)

            model = create_model(opt)
            model.setup()

            def _denoising_model(img_n):
                img_n = torch.Tensor(img_n.astype(np.float32)).unsqueeze(0)
                img_n = torch.reshape(img_n, (-1, img_n.shape[0], img_n.shape[1], img_n.shape[2]))
                model.set_input({'A': img_n})
                model.eval_mode()
                model.testing()
                img_n_dn = tensor2im(model.n_A_c)
                return img_n_dn


            # performance_evaluator(_denoising_model, opt.dir_noisy, save_results=True, dir_saving=dir_out_n)
            performance_evaluator2(_denoising_model, opt.dir_noisy, save_results=True, dir_saving=dir_out_n)
