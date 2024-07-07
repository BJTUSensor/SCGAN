
import itertools
from models.BaseNets import *
from models.BaseModel import BaseModel

from utils.util import *

class NoiseGAN(BaseModel):
    def __init__(self, opt):
        super(NoiseGAN, self).__init__(opt)
        self.name = 'NoiseGAN'
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        if not self.opt.cpu:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        else:
            self.device = torch.device('cpu', 0)

        # define the first networks
        self.netG_A = init_net(Generator(), self.gpu_ids)

        if self.isTrain:
            self.netD_A = init_net(Discriminator(), self.gpu_ids)

        # define loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionMatch = torch.nn.MSELoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.epsilon)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.epsilon)
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # specify the models you want to save to the disk.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A']
        else:
            self.model_names = ['G_A']

        self.loss_names = ['D_A', 'G_A', 'pn_map', 'clean_map', 'rec_map', 'noise_indp']
        self.visual_names = ['n_A', 'p_n', 'p_n_map_print', 'n_A_c'] + ['c_B', 'c_B_map_print', 'c_B_n',
                                                                        'c_B_n_map_print']
    def setup(self, log_file='.'):
        # load and print networks; create schedulers
        if self.isTrain:
            self.schedulers = [get_schedulers(optimizer, self.opt) for optimizer in self.optimizers]
            self.save_dir = os.path.join(self.opt.dir_checkpoints, self.opt.name)

            if not (self.opt.pre_train == '.'):
                self.load_networks(model_dir=self.opt.pre_train, which_epoch=self.opt.which_epoch)
            self.print_networks(verbose=False, log_file=log_file)
        else:
            self.load_networks(model_dir=self.opt.pre_train, which_epoch=self.opt.which_epoch)

    def set_input(self, x):
        self.n_A = x['A'].to(self.device)
        self.c_B = x['B'].to(self.device)


    def forward(self):
        self.n_A_map, self.n_A_c = self.netG_A(self.x_tp1.detach(), self.isTrain)

        # if not self.isTrain:
            # self.n_A_map *= torch.from_numpy(random.rand(1)).to(self.device)  #生成[0，1)之间的随机数
            # self.n_A_map *= random.randint(5,15)
        self.c_B_n = self.c_B + self.n_A_map

        if self.isTrain:
            self.p_n = self.n_A_map + 0.2
            self.p_n_map, _ = self.netG_A(self.p_n + 0.2)  #纯噪声-》纯噪声
            self.c_B_map, _ = self.netG_A(self.c_B + 0.2)  #干净图像-》0
            self.c_B_n_map, _ = self.netG_A(self.c_B_n + 0.2)  #重构图像 -》噪声
            # visualize noise map
            self.p_n_map_print = self.p_n_map.detach()
            self.c_B_map_print = self.c_B_map.detach()
            self.c_B_n_map_print = self.c_B_n_map.detach()

    def optimize_parameters(self, curr_step, curr_epoch):
        self.t = torch.randint(0, self.opt.num_timesteps, (self.n_A.size(0),), device=self.device)

        # epoch_phases = [30, 45, 60]
        ratio = self.opt.ratio_lambda
        if curr_epoch < self.opt.EP_lambda[0]:
            self.lambda_gan = self.opt.lambda_gan
            self.lambda_clean = self.opt.lambda_clean
            self.lambda_rec = self.opt.lambda_rec
            self.lambda_pn = self.opt.lambda_pn
        else:
            self.lambda_gan = self.opt.lambda_gan * ratio
            self.lambda_clean = self.opt.lambda_clean * ratio
            self.lambda_rec = self.opt.lambda_rec * ratio
            self.lambda_pn = self.opt.lambda_pn * 1

        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A], False)
        self.optimizer_G.zero_grad()
        self.cal_loss_G(curr_epoch)
        self.loss_G.backward()
        self.optimizer_G.step()

        # D_A and D_B


        if curr_step % self.opt.GD_rate == 0:
            self.set_requires_grad([self.netD_A], True)
            self.optimizer_D.zero_grad()
            self.cal_loss_D_A()
            self.loss_D_A.backward()
            self.optimizer_D.step()

    def cal_loss_G(self, curr_epoch):
        # GAN Loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.n_A_c), True) * self.lambda_gan  #降噪图像投入D，GANLoss

        if curr_epoch < self.opt.EP_loss[0]:
            self.loss_pn_map = 0
            self.loss_clean_map = 0
            self.loss_rec_map = 0
            self.loss_noise_indp = 0

        elif self.opt.EP_loss[0] <= curr_epoch < self.opt.EP_loss[1]:
            n_map_diff = self.n_A_map - self.p_n_map
            self.loss_pn_map = self.criterionMatch(n_map_diff,
                                                   torch.zeros_like(n_map_diff).to(self.device)) * self.lambda_pn
            self.loss_clean_map = self.criterionMatch(self.c_B_map, torch.zeros_like(self.c_B_map).to(
                self.device)) * self.lambda_clean
            self.loss_rec_map = 0
            self.loss_noise_indp = 0

        elif self.opt.EP_loss[1] <= curr_epoch < self.opt.EP_loss[2]:
            n_map_diff = self.n_A_map - self.p_n_map
            self.loss_pn_map = self.criterionMatch(n_map_diff,
                                                   torch.zeros_like(n_map_diff).to(self.device)) * self.lambda_pn
            self.loss_clean_map = self.criterionMatch(self.c_B_map, torch.zeros_like(self.c_B_map).to(
                self.device)) * self.lambda_clean

            n_map_diff = self.n_A_map - self.c_B_n_map
            self.loss_rec_map = self.criterionMatch(n_map_diff,
                                                    torch.zeros_like(n_map_diff).to(self.device)) * self.lambda_rec
            self.loss_noise_indp = 0

        else:
            n_map_diff = self.n_A_map - self.p_n_map
            self.loss_pn_map = self.criterionMatch(n_map_diff,
                                                   torch.zeros_like(n_map_diff).to(self.device)) * self.lambda_pn
            self.loss_clean_map = self.criterionMatch(self.c_B_map, torch.zeros_like(self.c_B_map).to(
                self.device)) * self.lambda_clean

            n_map_diff = self.n_A_map - self.c_B_n_map
            self.loss_rec_map = self.criterionMatch(n_map_diff,
                                                    torch.zeros_like(n_map_diff).to(self.device)) * self.lambda_rec

            diff_A_map = self.n_A_map[:, :, :, 1:] - self.n_A_map[:, :, :, :-1]
            var_diff = (self.n_A_map.var() - diff_A_map.var() * 0.5) * 50
            self.loss_noise_indp = self.criterionMatch(var_diff, torch.zeros_like(var_diff).to(self.device))

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_pn_map + self.loss_clean_map + self.loss_rec_map + self.loss_noise_indp

    def cal_loss_D_A(self):
        self.pred_real = self.netD_A(self.c_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        # Fake
        self.pred_fake = self.netD_A(self.n_A_c.detach())
        loss_D_fake = self.criterionGAN(self.pred_fake, False)
        # combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D_A = loss_D * self.lambda_gan
