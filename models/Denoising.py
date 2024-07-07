import itertools
from models.BaseModel import *
from models.ADNet import ADNet
from models.BRDNet import BRDNet
from models.DnCNN import DnCNN
from models.CTformer import CTformer
from models.BaseNets import *


class NoiseGAN(BaseModel):
    def __init__(self, opt):
        super(NoiseGAN, self).__init__(opt)
        self.name = 'NoiseGAN'
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.choice = opt.choice

        if not self.opt.cpu:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        else:
            self.device = torch.device('cpu', 0)

        # load/define networks
        ##此处做了修改
        if(self.choice == '1'):
           self.net = DnCNN()
        elif(self.choice == '2'):
            self.net = ADNet()
        elif(self.choice == '3'):
            self.net = BRDNet()
        elif(self.choice == '4'):
            self.net = CTformer()

        self.netG_A = init_net(self.net, self.gpu_ids)
        if self.isTrain:
            self.criterionMatch = torch.nn.MSELoss(reduction='sum').to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.epsilon)

            self.optimizers = [self.optimizer_G]

        self.model_names = ['G_A']
        self.loss_names = ['match']
        self.visual_names = ['n_A', 'n_A_map', 'n_A_c'] + ['c_B']

    def setup(self, log_file='.'):
        if self.isTrain:
            self.schedulers = [get_schedulers(optimizer, self.opt) for optimizer in self.optimizers]
            self.save_dir = os.path.join(self.opt.dir_checkpoints, self.opt.name)
            if not (self.opt.pre_train == '.'):
                self.load_networks(self.opt.pre_train, self.opt.which_epoch)
            self.print_networks(verbose=False, log_file=log_file)
        else:
            self.load_networks(model_dir=self.opt.pre_train, which_epoch=self.opt.which_epoch)

    def set_input(self, x):
        if self.isTrain:
            self.n_A = x['A'].to(self.device)
            self.c_B = x['B'].to(self.device)
        else:
            self.n_A = x['A'].to(self.device)

    def forward(self):
        if self.isTrain:
            self.n_A_map, self.n_A_c = self.netG_A(self.n_A, isTrain=True)
            self.n_A_map = 0.2 + self.n_A_map
        else:
            self.n_A_map, self.n_A_c = self.netG_A(self.n_A, isTrain=False)

    def cal_loss_G(self, curr_epoch):
        self.loss_match = self.criterionMatch(self.n_A_c, self.c_B) * 0.5
        self.loss_G = self.loss_match

    def optimize_parameters(self, curr_step, curr_epoch):
        self.forward()
        self.set_requires_grad([self.netG_A], True)
        self.optimizer_G.zero_grad()
        self.cal_loss_G(curr_epoch)
        self.loss_G.backward()
        self.optimizer_G.step()
