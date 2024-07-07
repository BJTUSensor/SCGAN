import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, img_channels=1):
        super(DnCNN, self).__init__()
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
            layers.append(nn.BatchNorm2d(num_features=n_channels,
                                         eps=0.0001,
                                         momentum=0.95))
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
            out = out - out.mean()
        return out, x - out