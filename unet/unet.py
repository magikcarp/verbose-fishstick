import torch
import torch.nn as nn


class UNet(nn.Module):
    """ UNet network that expects input of 512x512 and produces a mask. 
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Encoding arm
        self.down_conv = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in
              [(in_channels, 64), (64, 128), (128, 256), (256, 512)]]
        )
        self.down_sample = nn.ModuleList(
            [DownSample() for _ in range(4)]
        )

        # Bottom of the U
        self.middle_conv = DoubleConvolution(512, 1024)

        # Decoding arm
        self.up_conv = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in
              [(1024, 512), (512, 256), (256, 128), (128, 64)]]
        )
        self.up_sample = nn.ModuleList(
            [UpSample(i, o) for i, o in
              [(1024, 512), (512, 256), (256, 128), (128, 64)]]
        )
        self.concat = nn.ModuleList([torch.cat() for _ in range(4)])

        # Output
        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def foward(self, x: torch.Tensor):
        through_pass = []
        for i in range(len(self.down_conv)):
            _x = self.down_conv[i](x)
            through_pass.append(_x)
            _x = self.down_sample[i](_x)
        _x = self.middle_conv(_x)
        for i in range(len(self.up_conv)):
            _x = self.up_sample[i](_x)
            _x = self.concat[i](_x, through_pass.pop())
            _x = self.up_conv[i](_x)
        _x = self.last_conv(_x)
        _x = self.sigmoid(_x)
        return _x


class DoubleConvolution(nn.Module):
    """ Neural network layer that performs two convolutions with a 3x3 kernel,
        each followed by ReLU activation. 
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.a1 = nn.ReLU()
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.a2 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        _x = self.c1(x)
        _x = self.a1(_x)
        _x = self.c2(_x)
        _x = self.a2(_x)
        return _x


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


