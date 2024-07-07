import importlib
import torch
import torch.utils.data


def CreateDataLoader(opt):
    Data = CustomDatasetDataLoader(opt)
    return Data


class CustomDatasetDataLoader(object):
    """docstring for CustomDatasetDataLoader"""

    def __init__(self, opt):
        super(CustomDatasetDataLoader, self).__init__()
        self.opt = opt
        self.name = 'CustomDatasetDataLoader'

        module = importlib.import_module('dataloader.' + opt.dataset)

        self.dataset = getattr(module, opt.dataset)(opt)  #getattr函数的作用是从一个对象中获取指定名称的属性。可以通过getattr(object, name)来获取对象object中名为name的属性。如果该属性不存在，则会抛出一个异常
        self.opt.steps_per_epoch = len(self.dataset) / opt.batch_size
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True)

    def name(self):
        return self.name
