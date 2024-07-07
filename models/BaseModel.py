import os
from collections import OrderedDict

import torch


class BaseModel():
    def __init__(self, opt):
        pass

    def set_input(self, x):
        pass

    def setup(self):
        pass

    def forward(self):
        pass

    def testing(self):
        with torch.no_grad():
            self.forward()

    def eval_mode(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def load_networks(self, model_dir, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(model_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                # for k, v in state_dict.items():
                #     # 手动添加“module.”
                #     if 'module' not in k:
                #         k = 'module.' + k
                #     else:
                #         # 调换module和features的位置
                #         k = k.replace('module.', '')
                #     new_state_dict[k] = v

                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    new_state_dict[k] = v

                for key in list(new_state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(new_state_dict)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
                if module.__class__.__name__.startswith('InstanceNorm') and \
                        (key == 'running_mean' or key == 'running_var'):
                    if getattr(module, key) is None:
                        state_dict.pop('.'.join(keys))
                if module.__class__.__name__.startswith('InstanceNorm') and \
                        (key == 'num_batches_tracked'):
                    state_dict.pop('.'.join(keys))
            else:
                self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # print the network information
    def print_networks(self, verbose, log_file='.'):
        print('------------------ Networks initilized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                if log_file != '.':
                    f = open(log_file, 'a')
                    print('====' + 'net' + name + '====', file=f)
                    print(net, file=f)
                    f.close()
                print('[Network %s] Total number of parameters: %.3f M' % (name, num_params / 1e6))
        print('----------------------------------------------------')

    def optimize_parameters(self):
        pass

    def update_learning_rate(self, visualizer):
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        message = 'learning rate = %.7f' % lr
        print(message)
        visualizer.print_message_in_log(message)

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                net = getattr(self, 'net' + name)

                if torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.opt.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
            return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            print(name)
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
