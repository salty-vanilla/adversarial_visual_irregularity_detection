import functools
from torchvision import datasets, transforms
import torch
from PIL import Image
import os
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, target_size=None,
                 is_flip=False,
                 *args, 
                 **kwargs):
        self.transform = []
        if target_size is not None:
            if not isinstance(target_size, int):
                target_size = (target_size[1], target_size[0])
            self.transform.append(transforms.Resize(target_size))
        if is_flip:
            self.transform.append(transforms.RandomHorizontalFlip())
        self.transform.append(transforms.ToTensor())
        self.transform.append(transforms.Lambda(lambda x: (x - 0.5)*2))
        self.transform = transforms.Compose(self.transform)
        self._length = 0
        self._get_item = None

    def flow(self, x, 
             y=None, 
             batch_size=32,
             shuffle=True,
             *args,
             **kwargs):
        def _get_item(index):
            image = x[index]
            label = y[index] if y is not None else None
            image = self.transform(image)
            if label is None:
                return image
            else:   
                return image, label

        self._get_item = _get_item
        self._length = len(x)
        return torch.utils.data.DataLoader(self, batch_size, 
                                           shuffle,
                                           **kwargs)

    def flow_from_directory(self, image_dir, 
                            with_labels=False, 
                            batch_size=32,
                            shuffle=True,
                            *args,
                            **kwargs):
        if with_labels:
            dirs = [os.path.join(image_dir, f)
                    for f in os.listdir(image_dir)
                    if os.path.isdir(os.path.join(image_dir, f))]
            dirs = sorted(dirs)
            image_paths = [get_image_paths(d) for d in dirs]
            labels = []
            for i, ip in enumerate(image_paths):
                labels += [i] * len(ip)
            labels = np.array(labels)
            image_paths = np.array(functools.reduce(lambda x, y: x+y, image_paths))
        else:
            image_paths = np.array([path for path in get_image_paths(image_dir)])
            labels = None
        return self.flow_from_paths(image_paths, 
                                    labels,
                                    batch_size,
                                    shuffle,
                                    *args,
                                    **kwargs)

    def flow_from_paths(self, paths, 
                        labels=None,
                        batch_size=32,
                        shuffle=True,
                        *args,
                        **kwargs):
        def _get_item(index):
            path = paths[index]
            label = labels[index] if labels is not None else None
            image = Image.open(path)
            image = self.transform(image)
            if label is None:
                return image
            else:
                return image, label
        self._get_item = _get_item
        self._length = len(paths)
        return torch.utils.data.DataLoader(self, batch_size, 
                                           shuffle,
                                           **kwargs)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._get_item(index)


def get_image_paths(src_dir):
    def get_all_paths():
        for root, dirs, files in os.walk(src_dir):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def is_image(path):
        _, ext = os.path.splitext(path)
        ext = ext[1:]
        if ext in ['png', 'jpg', 'bmp']:
            return True
        else:
            return False

    return [path for path in get_all_paths() if is_image(path)]

