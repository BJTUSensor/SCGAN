import argparse

parser = argparse.ArgumentParser(description='Self-consistent GAN')

# Mode
parser.add_argument('--isTrain', type=bool, default=True, choices=(True, False),
                    help='training mode or evaluating mode')
parser.add_argument('--name', type=str, default='small2')
parser.add_argument('--choice',type=str,default='4')
# Hardware
parser.add_argument('--cpu', action='store_true',
                    help='use only cpu')
parser.add_argument('--gpu_ids', type=list, default=[0])
parser.add_argument('--seed', type=int, default=3, help='random seed')

# Model
parser.add_argument('--mode', type=str, default='Denoising', choices=('NoiseModeling', 'Denoising'))

# Data
parser.add_argument('--n_colors', type=int, default=1)
parser.add_argument('--dir_noisy', type=str, default='data/real/train/simulate_noisyA')
parser.add_argument('--dir_clean', type=str, default='data/real/train/cleanA')
parser.add_argument('--pre_train', type=str, default='.')
parser.add_argument('--dn_methods', type=str, default='SCGAN')

## test
parser.add_argument('--save_results', action='store_true', help='save output results')

opt, _ = parser.parse_known_args()

if True:
    # ******************************
    # Training / Testing configuration
    ## loss objective
    parser.add_argument('--loss', type=str, default='GANLoss', help='loss function configuration')
    parser.add_argument('--GD_rate', type=int, default=5, help='5 G, 1 D') ######
    parser.add_argument('--lambda_gan', type=float, default=2)
    parser.add_argument('--lambda_rec', type=float, default=2)
    parser.add_argument('--lambda_pn', type=float, default=2)
    parser.add_argument('--lambda_clean', type=float, default=2)
    parser.add_argument('--EP_lambda', type=list, default=[1000], help='change the weights at epoch #')
    parser.add_argument('--ratio_lambda', type=int, default=1.5)
    # parser.add_argument('--EP_loss', type=list, default=[5, 8, 17], help='GAN Loss, + pure noise & clean, + noise rec')
    parser.add_argument('--EP_loss', type=list, default=[75, 30, 50],
                        help='GAN Loss, + pure noise & clean, + noise rec')   #######
    ## data
    parser.add_argument('--dataset', type=str, default='TwoFolders')
    parser.add_argument('--dir_data', type=str, default='data/real/train/')
    parser.add_argument('--dataset_noisy', type=str, default='noisyD')
    parser.add_argument('--dataset_clean', type=str, default='cleanA')
    parser.add_argument('--patch_size', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)

    ## optimizer
    parser.add_argument('--optimizer', default='Adam', choices=('Adam', 'SGD', 'RMSprop'))
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Adam epsilon for numerical stability')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--lr_policy', type=str, default='step', help='learning decay type')
    parser.add_argument('--steps_per_epoch', type=int, default=int(50),
                        help='# of steps/batches each epoch, len(dataloader)')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--which_epoch', type=int, default=0)
    parser.add_argument('--lr_decay_epochs', type=int, default=80, help='learning rate decay per N epochs')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='how many batches to wait before logging training status')
    ## store
    parser.add_argument('--dir_checkpoints', type=str, default='./checkpoints/', help='checkpoints dir, saving models')

    # utilities
    parser.add_argument('--display_winsize', type=int, default=200)
    parser.add_argument('--display_id', type=int, default=-1)
    parser.add_argument('--display_server', type=str, default="http://localhost")
    parser.add_argument('--display_port', type=int, default=8097)
    parser.add_argument('--display_env', type=str, default='main')
    parser.add_argument('--display_ncols', type=int, default=8)
    parser.add_argument('--no_html', action='store_true')

    opt = parser.parse_args()
