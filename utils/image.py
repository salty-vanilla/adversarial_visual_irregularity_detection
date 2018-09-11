import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def tile_images(dst_path, x,
                transform=lambda x: (x+1)/2 * 255):
    n = int(np.sqrt(len(x)))
    x = x[:n ** 2]
    x = np.transpose(x, (0, 2, 3, 1))
    h, w, c = x.shape[1:]
    x = x.reshape(n, n, *x.shape[1:])
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(n * h, n * w, c)
    if c == 1:
        x = np.squeeze(x, -1)
    x = transform(x)
    x = x.numpy().astype('uint8')
    image = Image.fromarray(x)
    image.save(dst_path)


def tile_results(dst_dir,
                 x, x_eta, x_fake, diff, d_x):
    os.makedirs(dst_dir, exist_ok=True)
    n, c, h, w = x.shape
    cmap = None

    if c == 1:
        x = np.squeeze(x, 1)
        x_eta = np.squeeze(x_eta, 1)
        x_fake = np.squeeze(x_fake, 1)
        cmap= 'gray'

    for i in range(n):
        dst_path = os.path.join(dst_dir, '%d.png' % i)
        plt.figure()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplot(151)
        plt.imshow(x[i], vmin=-1., vmax=1., cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(152)
        plt.imshow(x_eta[i], vmin=-1., vmax=1., cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(153)
        plt.imshow(x_fake[i], vmin=-1., vmax=1., cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(154)
        plt.imshow(diff[i], vmin=0, vmax=2., cmap='jet')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(155)
        plt.imshow(d_x[i], vmin=0, vmax=1., cmap='jet')
        plt.xticks([])
        plt.yticks([])

        plt.savefig(dst_path, bbox_inches='tight', pad_inches=0., dpi=300)
