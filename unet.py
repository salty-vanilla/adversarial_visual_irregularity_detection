import torch
from blocks import ConvBlock


class DownModule(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch,
                              out_ch)

    def forward(self, x):
        x = self.conv(x)
        return torch.nn.MaxPool2d(2, 2)(x), x


class UpModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch,
                              out_ch,
                              sampling='upsampling')

    def forward(self, x1, x2):
        x = self.conv(x1)
        return torch.cat([x, x2], dim=1)


class DownBlock(torch.nn.Module):
    def __init__(self, *channels):
        super().__init__()
        self.convs = torch.nn.Sequential(*[ConvBlock(channels[i],
                                                     channels[i+1],
                                                     normalization='batch')
                                           for i in range(len(channels)-2)])
        self.down = DownModule(channels[-2], channels[-1])

    def forward(self, x):
        x = self.convs(x)
        return self.down(x)


class UpBlock(torch.nn.Module):
    def __init__(self, *channels):
        super().__init__()
        self.convs = torch.nn.Sequential(*[ConvBlock(channels[i],
                                                     channels[i+1],
                                                     normalization='batch')
                                           for i in range(len(channels)-2)])
        self.up = UpModule(channels[-2], channels[-1])

    def forward(self, x1, x2):
        x = x1
        x = self.convs(x)
        return self.up(x, x2)


class UNet(torch.nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.down1 = DownBlock(in_ch, 64, 64)
        self.down2 = DownBlock(64, 128, 128)
        self.down3 = DownBlock(128, 256, 256, 256)
        self.down4 = DownBlock(256, 512, 512, 512)
        self.down5 = DownBlock(512, 512, 512, 512)

        self.up1 = UpBlock(512, 512, 512, 256)
        self.up2 = UpBlock(512+256, 512, 256)
        self.up3 = UpBlock(256+512, 512, 128)
        self.up4 = UpBlock(128+256, 256, 64)
        self.up5 = UpBlock(64+128, 128, 32)
        self.last_conv = ConvBlock(32+64, in_ch,
                                   activation=torch.nn.Tanh())

    def forward(self, x):
        x, d1 = self.down1(x)
        x, d2 = self.down2(x)
        x, d3 = self.down3(x)
        x, d4 = self.down4(x)
        x, d5 = self.down5(x)

        x = self.up1(x, d5)
        x = self.up2(x, d4)
        x = self.up3(x, d3)
        x = self.up4(x, d2)
        x = self.up5(x, d1)

        return self.last_conv(x)


if __name__ == '__main__':
    import numpy as np
    inpt = torch.Tensor(np.random.normal(size=(1, 3, 224, 224)))
    unet = UNet(3)
    o = unet(inpt)
    print(o.size())



