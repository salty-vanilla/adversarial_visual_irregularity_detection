import torch
import yaml
import sys
import os
from unet import UNet
from discriminator import Discriminator
from data_loader import Dataset
from predictor import Predictor


yml_path = sys.argv[1]
with open(yml_path) as f:
    config = yaml.load(f)

if config['use_gpu']:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

discriminator = Discriminator(**(config['discriminator_params']))
unet = UNet(**(config['unet_params']))

dl = Dataset(**(config['dataset_params'])) \
    .flow_from_directory(**(config['test_dataloader_params']))

unet_path = os.path.join(config['fit_params']['logdir'],
                         'unet_%d.pth' % config['test_epoch'])
unet.load_state_dict(torch.load(unet_path))

discriminator_path = os.path.join(config['fit_params']['logdir'],
                                  'discriminator_%d.pth' % config['test_epoch'])
discriminator.load_state_dict(torch.load(discriminator_path))

p = Predictor(unet, discriminator)
p(dl, os.path.join(config['fit_params']['logdir'], 'predicted'))