import numpy as np
import os
import ntpath
from utils import util
from utils import html
import cv2
import scipy.io as scio
from options import opt


# Save image to the disk
# this class can be used for /test/ phase.
def save_images(webpage, visuals, image_path, width=200):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data, opt)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)

        min_v = image_numpy.min()
        max_v = image_numpy.max()
        image_numpy = (image_numpy - min_v) / (max_v - min_v) * 255

        cv2.imwrite(save_path, im)
        scio.savemat(os.path.join(image_dir, '%s_%s.mat' % (name, label)), {'data': im})

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


def norm(data):
    min_v = data.min()
    max_v = data.max()
    data = (data - min_v) / (max_v - min_v) * 255
    return data


# Do "python -m visdom.server" before runing scripts
class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        # Yes/Not: use real-time web visualization
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env,
                                     raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(opt.dir_checkpoints, opt.name)
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.dir_checkpoints, opt.name, 'config.txt')
        with open(self.log_name, "a") as f:
            f.write('================ Config ================\n')
            for arg in vars(opt):
                f.write('{}: {}\n'.format(arg, getattr(opt, arg)))
            f.write('\n')
            f.write('================ Net Summary ================\n')

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print(
            '\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image, opt)
                    # if img is gray, then expand the dim
                    if image_numpy.ndim == 2:
                        image_numpy = np.expand_dims(image_numpy, axis=2)

                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except ConnectionError:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image, opt)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image, opt)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                # util.save_image(image_numpy, img_path)\
                image_numpy = norm(image_numpy)
                cv2.imwrite(img_path, image_numpy)
                scio.savemat(os.path.join(self.img_dir, 'epoch%.3d_%s.mat' % (epoch, label)), {'data': image_numpy})
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, _ in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, losses):
        print('plot')
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            X_coord = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
            Y_coord = np.array(self.plot_data['Y'])
            # there is bug of visdom: if X or Y is a vector, then ndim has to be 1.
            if len(self.plot_data['legend']) == 1:
                X_coord = X_coord.reshape(-1)
                Y_coord = Y_coord.reshape(-1)

            assert X_coord.shape == Y_coord.shape
            self.vis.line(
                X=X_coord,
                Y=Y_coord,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, steps_per_epoch, losses, t_total, t_data, t_model):
        message = 'Epoch: [%d], steps: [%d/%d], t/log: [%.3f], [data: %.3f, model: %.3f] >>> ' % (
        epoch, i, steps_per_epoch, t_total, t_data, t_model)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_message_in_log(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
