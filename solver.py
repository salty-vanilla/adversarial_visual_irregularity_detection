import torch
import torch.utils.data
import os
import time
from PIL import Image
import numpy as np
from utils.image import tile_results


class Solver:
    def __init__(self, unet: torch.nn.Module,
                 discriminator: torch.nn.Module,
                 sigma=1.,
                 device='cuda'):
        self.device = device
        self.unet = unet.to(device)
        self.discriminator = discriminator.to(device)
        self.unet.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)
        self.z_sampler = torch.distributions.Normal(0., sigma)

    def fit(self, data_loader: torch.utils.data.DataLoader,
            nb_epoch: int = 100,
            lr_d: float = 2e-4,
            lr_g: float = 2e-4,
            logdir: str = 'logs',
            save_steps: int = 10,
            visualize_steps: int = 1):
        self.discriminator.train()
        self.unet.train()

        os.makedirs(logdir, exist_ok=True)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr_d,
                                 betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.unet.parameters(), lr_g,
                                 betas=(0.5, 0.999))

        criterion = SquaredBCE()
        # criterion = criterion.to(self.device)

        for epoch in range(1, nb_epoch + 1):
            print('\nEpoch %d / %d' % (epoch, nb_epoch))
            start = time.time()
            for iter_, x in enumerate(data_loader):
                bs = x.shape[0]
                # update discriminator
                opt_d.zero_grad()
                x_real = x.to(self.device)
                x_eta = x_real + self.z_sampler.sample(x.shape).to(self.device)
                x_eta = torch.clamp(x_eta, -1., 1.)
                x_fake = self.unet(x_eta)

                d_x_real = self.discriminator(x_real)
                t_real = torch.full((bs, 1, 1, 1), 1,
                                    device=self.device)
                d_x_fake = self.discriminator(x_fake)
                t_fake = torch.full((bs, 1, 1, 1), 0,
                                    device=self.device)
                loss_d = criterion(d_x_real, t_real) + criterion(d_x_fake, t_fake)
                loss_d.backward()
                opt_d.step()

                # update generator
                opt_g.zero_grad()
                x_eta = x_real + self.z_sampler.sample(x.shape).to(self.device)
                x_fake = self.unet(x_eta)
                d_x_fake = self.discriminator(x_fake)
                loss_g = criterion(d_x_fake, t_real)
                loss_g.backward()
                opt_g.step()

                print('%d / %d' % (iter_, len(data_loader)),
                      '%.1f[s]  ' % (time.time() - start),
                      'loss_d: %.3f' % loss_d.item(),
                      'loss_g: %.3f' % loss_g.item(),
                      end='\r')

            if epoch % save_steps == 0:
                torch.save(self.unet.state_dict(),
                           os.path.join(logdir, 'unet_%d.pth' % epoch))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(logdir, 'discriminator_%d.pth' % epoch))

            if epoch % visualize_steps == 0:
                diff = (x_real - x_fake) ** 2
                d_x = d_x_fake

                diff = torch.mean(diff, dim=1)
                d_x = torch.squeeze(d_x, dim=1)

                x_real = x_real.detach().cpu().numpy()
                x_eta = x_eta.cpu().numpy()
                x_fake = x_fake.detach().cpu().numpy()
                diff = diff.detach().cpu().numpy()
                d_x = d_x.detach().cpu().numpy()

                tile_results(os.path.join(logdir, 'epoch_%d' % epoch),
                             x_real, x_eta, x_fake, diff, d_x)

    def init_weights(self, m):
        is_init = isinstance(m, torch.nn.Conv2d) \
                  or isinstance(m, torch.nn.ConvTranspose2d) \
                  or isinstance(m, torch.nn.Linear)
        if is_init:
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class SquaredBCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, t, eps=1e-8):
        log = lambda x: torch.log(torch.clamp(x, eps, 1.))
        return -(t*log(y**2) + (1-t)*log((1-y**2))).mean()
