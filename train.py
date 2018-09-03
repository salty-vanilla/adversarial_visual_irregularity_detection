import sys
import yaml
from solver import Solver
from unet import UNet
from discriminator import Discriminator


yml_path = sys.argv[1]
with open(yml_path) as f:
    config = yaml.load(f)

discriminator = Discriminator(**(config['discriminator_params']))
unet = UNet(**(config['unet_params']))

solver = Solver(unet, discriminator)

solver.fit(**(config['fit_params']))