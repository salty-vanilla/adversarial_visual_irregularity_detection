import torch
import sys
import yaml
from solver import Solver
from unet import UNet
from discriminator import Discriminator
from data_loader import Dataset

yml_path = sys.argv[1]
with open(yml_path) as f:
    config = yaml.load(f)

if config['use_gpu']:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

discriminator = Discriminator(**(config['discriminator_params']))
unet = UNet(**(config['unet_params']))

data_loader = Dataset(**(config['dataset_params']))\
    .flow_from_directory(**(config['dataloader_params']))

solver = Solver(unet, discriminator)
solver.fit(data_loader, **(config['fit_params']))
