import numpy as np
from PIL import Image


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
