import torch


class SubpixelConv(torch.nn.Conv2d):
    def __init__(self, in_ch,
                 out_ch=None,
                 kernel_size=3,
                 rate=2):
        if out_ch is None:
            out_ch = in_ch * (rate**2)
        else:
            out_ch = out_ch * (rate**2)

        super().__init__(in_ch, out_ch,
                         kernel_size=kernel_size,
                         padding=kernel_size//2,
                         bias=False)

        self.ps = torch.nn.PixelShuffle(rate)

    def forward(self, x):
        return self.ps(super().forward(x))
