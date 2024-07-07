import torch.nn as nn
from torch.optim import lr_scheduler
from utils.functions import *


def get_schedulers(optimizier, opt):
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizier, step_size=opt.lr_decay_epochs, gamma=opt.gamma)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, gpu_ids, init_gain=0.02):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        print('1111')
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_gain)
    return net


def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network')
    net.apply(init_func)


#Generator
class Generator(nn.Module):
    def __init__(self, depth=17, n_channels=64, img_channels=1):
        super(Generator, self).__init__()
        layers = []
        kernel_size = 3
        padding = 1
        padding_mode = 'circular'
        layers.append(nn.Conv2d(in_channels=img_channels,
                                out_channels=n_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                padding_mode=padding_mode,
                                bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels,
                                    out_channels=n_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    padding_mode=padding_mode,
                                    bias=False))
            # layers.append(nn.BatchNorm2d(num_features=n_channels,
            #                              eps=0.0001,
            #                              momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels,
                                out_channels=img_channels,
                                kernel_size=kernel_size,
                                padding_mode=padding_mode,
                                padding=padding,
                                bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x, isTrain=True):
        out = self.dncnn(x)
        if not isTrain:
            out = out - out.mean()  #对输出的噪声0均值归一化，均值为整个矩阵的均值
        return out, x - out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        m_head = []
        kernel_size = 5
        in_channels = 1
        out_channels = 64
        stride = 2
        m_head.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=0,
                                bias=True))
        m_head.append(nn.LeakyReLU(negative_slope=0.2))

        kernel_size = 5
        in_channels = out_channels
        out_channels = 128
        stride = 2
        m_head.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=0,
                                bias=True))
        m_head.append(nn.LeakyReLU(negative_slope=0.2))

        m_tail = []
        kernel_size = 3
        in_channels = out_channels
        out_channels = 64
        stride = 1
        m_tail.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=0,
                                bias=True))
        m_tail.append(nn.LeakyReLU(negative_slope=0.2))

        kernel_size = 3
        in_channels = out_channels
        out_channels = 1
        stride = 1
        m_tail.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=0,
                                bias=True))
        m_tail.append(nn.LeakyReLU(negative_slope=0.2))

        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.tail(x)
        return x




class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
