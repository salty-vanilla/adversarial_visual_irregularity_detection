import torch
from layers.gan import WeightEqualizer
from layers.normalization import PixelNorm, LayerNorm
from layers.conv import SubpixelConv


class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch,
                 out_ch,
                 kernel_size=3,
                 sampling='same',
                 padding='same',
                 use_bias=True,
                 normalization=None,
                 activation=torch.nn.LeakyReLU(0.2),
                 dropout_rate=0.0,
                 is_weight_equalizer=True):
        assert sampling in ['deconv', 'subpixel', 'upsampling',
                            'stride', 'max_pool', 'avg_pool',
                            'same']
        assert normalization in ['batch', 'layer', 'pixel', 'instance', None]
        super().__init__()

        # upsampling
        self.net = torch.nn.Sequential()

        if sampling == 'upsampling':
            self.net.add_module('up', torch.nn.UpsamplingNearest2d(scale_factor=2))

        # convolution
        padding = kernel_size // 2 if padding == 'same' else 0
        if sampling in ['same', 'max_pool', 'avg_pool', 'upsampling']:
            conv = torch.nn.Conv2d(in_ch,
                                   out_ch,
                                   kernel_size,
                                   stride=1,
                                   padding=padding,
                                   bias=use_bias)
        elif sampling == 'stride':
            conv = torch.nn.Conv2d(in_ch,
                                   out_ch,
                                   kernel_size,
                                   stride=2,
                                   padding=padding,
                                   bias=use_bias)
        elif sampling == 'deconv':
            conv = torch.nn.ConvTranspose2d(in_ch,
                                            out_ch,
                                            kernel_size,
                                            stride=2,
                                            padding=padding,
                                            bias=use_bias)
        elif sampling == 'subpixel':
            conv = SubpixelConv(in_ch,
                                rate=2)
        else:
            raise ValueError
        self.net.add_module('conv', conv)

        if is_weight_equalizer:
            self.net.add_module('weight_equalizer', WeightEqualizer(conv))

        # normalization
        if normalization is not None:
            if normalization == 'batch':
                norm = torch.nn.BatchNorm2d(out_ch)
            elif normalization == 'layer':
                norm = LayerNorm(out_ch)
            elif normalization == 'pixel':
                norm = PixelNorm()
            elif normalization == 'instance':
                norm = torch.nn.InstanceNorm2d(out_ch)
            else:
                raise ValueError
            self.net.add_module('norm', norm)

        # activation
        if activation is not None:
            self.net.add_module('act', activation)

        # dropout
        if dropout_rate != 0.0:
            self.net.add_module('dropout', torch.nn.Dropout2d(dropout_rate))

        # pooling
        if sampling == 'max_pool':
            self.net.add_module('pool', torch.nn.MaxPool2d(2, 2))
        elif sampling == 'avg_pool':
            self.net.add_module('pool', torch.nn.AvgPool2d(2, 2))
        else:
            pass

    def forward(self, x):
        return self.net(x)
