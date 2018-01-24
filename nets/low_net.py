import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, inner_chanels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(),
            nn.Conv2d(in_channels,
                      inner_chanels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False
                      ),
            nn.BatchNorm2d(inner_chanels),
            nn.ELU(),
            nn.Conv2d(inner_chanels,
                      inner_chanels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False
                      ),
            nn.BatchNorm2d(inner_chanels),
            nn.ELU(),
            nn.Conv2d(inner_chanels,
                      in_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      ),
        )

    def forward(self, x):
        return x + self.bottleneck(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels,
                                            out_channels*4,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            ),
                                  nn.PixelShuffle(2)
                                  )

    def forward(self, x):
        return self.conv(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_block32 = nn.Sequential(ResBlock(65, 32),
                                     ResBlock(65, 32),
                                     ResBlock(65, 32),
                                     DownBlock(65, 128))

        self.down_block16 = nn.Sequential(ResBlock(128, 32),
                                     ResBlock(128, 32),
                                     ResBlock(128, 32),
                                     #ResBlock(128, 32),
                                     DownBlock(128, 256))


        self.block8 = nn.Sequential(ResBlock(256, 64),
                                        ResBlock(256, 64),
                                        #ResBlock(256, 64),
                                        #ResBlock(256, 64),
                                        #ResBlock(256, 64),
                                        UpBlock(256, 256)
                                        )

        self.up_block16 = nn.Sequential(ResBlock(256, 64),
                                        ResBlock(256, 64),
                                        #ResBlock(256, 64),
                                        #ResBlock(256, 64),
                                        UpBlock(256, 128)
                                        )

        self.up_block32 = nn.Sequential(ResBlock(128, 32),
                                        ResBlock(128, 32),
                                        #ResBlock(128, 32),
                                        #ResBlock(128, 32),
                                        UpBlock(128, 128),
                                        )
        self.up_block64 = nn.Sequential(ResBlock(128, 32),
                                        ResBlock(128, 32),
                                        #ResBlock(128, 32),
                                        #ResBlock(128, 32),
                                        )
        self.result = nn.Conv2d(128,
                                30,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False
                                )

        self.net = nn.Sequential(#self.down_block64,
                                 self.down_block32,
                                 self.down_block16,
                                 self.block8,
                                 self.up_block16,
                                 self.up_block32,
                                 self.up_block64,
                                 self.result,
                                 nn.ReLU(),
                                 )

    def forward(self, x):
        return self.net(x)