import torch
from blocks import ConvBlock


class Discriminator(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 96,
                                kernel_size=11,
                                sampling='stride',
                                normalization='batch',
                                stride=4,
                                padding='valid')
        self.block2 = torch.nn.Sequential(ConvBlock(96, 96,
                                                    kernel_size=1,
                                                    normalization='batch'),
                                          ConvBlock(96, 96,
                                                    kernel_size=1,
                                                    normalization='batch'),
                                          ConvBlock(96, 96,
                                                    kernel_size=3,
                                                    normalization='batch',
                                                    sampling='stride',
                                                    padding='valid'))

        self.block3 = torch.nn.Sequential(ConvBlock(96, 96,
                                                    kernel_size=1,
                                                    normalization='batch'),
                                          ConvBlock(96, 96,
                                                    kernel_size=1,
                                                    normalization='batch'),
                                          ConvBlock(96, 256,
                                                    kernel_size=5,
                                                    normalization='batch',
                                                    sampling='stride',
                                                    padding='valid'))

        self.block4 = torch.nn.Sequential(ConvBlock(256, 512,
                                                    kernel_size=1,
                                                    normalization='batch'),
                                          ConvBlock(512, 1024,
                                                    kernel_size=1,
                                                    normalization='batch'),
                                          ConvBlock(1024, 1024,
                                                    kernel_size=1,
                                                    normalization='batch'),
                                          ConvBlock(1024, 1,
                                                    kernel_size=1,
                                                    activation=torch.nn.Sigmoid()))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)


if __name__ == '__main__':
    import numpy as np
    inpt = torch.Tensor(np.random.normal(size=(1, 3, 224, 224)))
    d = Discriminator(3)
    o = d(inpt)
    print(o.size())