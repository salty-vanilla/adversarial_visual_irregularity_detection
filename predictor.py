import torch.utils.data
import os
from utils.image import tile_results


class Predictor:
    def __init__(self, unet: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 device='cuda'):
        self.device = device
        self.unet = unet.to(device)
        self.discriminator = discriminator.to(device)
        self.z_sampler = torch.distributions.Normal(0., 0.5)

    def __call__(self, data_loader: torch.utils.data.DataLoader,
                 logdir: str,
                 nb_visualize_batch: int = 1):
        os.makedirs(logdir, exist_ok=True)
        self.discriminator.eval()
        self.unet.eval()
        for iter_, x in enumerate(data_loader):
            x_real = x.to(self.device)
            x_eta = x_real + self.z_sampler.sample(x.shape).to(self.device)
            x_eta = torch.clamp(x_eta, -1., 1.)
            x_fake = self.unet(x_eta)
            diff = (x_real - x_fake) ** 2
            d_x = self.discriminator(x_real)

            diff = torch.mean(diff, dim=1)
            d_x = torch.squeeze(d_x, dim=1)

            if iter_ < nb_visualize_batch:
                x_real = x_real.detach().cpu().numpy()
                x_eta = x_eta.cpu().numpy()
                x_fake = x_fake.detach().cpu().numpy()
                diff = diff.detach().cpu().numpy()
                d_x = d_x.detach().cpu().numpy()

                tile_results(os.path.join(logdir, 'batch_%d' % iter_),
                             x_real, x_eta, x_fake, diff, d_x)


if __name__ == '__main__':
    import sys
    import yaml
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

    dl = Dataset(**(config['dataset_params'])) \
        .flow_from_directory(**(config['dataloader_params']))

    unet.load_state_dict(torch.load('../logs/debug/unet_20.pth'))
    discriminator.load_state_dict(torch.load('../logs/debug/discriminator_20.pth'))

    p = Predictor(unet, discriminator)

    p(dl, './temp')
