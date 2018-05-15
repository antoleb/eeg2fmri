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
    
    
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, inner_chanels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ELU(),
            nn.Conv3d(in_channels,
                      inner_chanels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      ),
            nn.BatchNorm3d(inner_chanels),
            nn.ELU(),
            nn.Conv3d(inner_chanels,
                      inner_chanels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.BatchNorm3d(inner_chanels),
            nn.ELU(),
            nn.Conv3d(inner_chanels,
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
                              kernel_size=(3, 6),
                              stride=(2, 4),
                              padding=1,
                              )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, 3, 2, 1, 1)

    def forward(self, x):
        return self.conv(x)

    
class Reshaper(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.view(-1, 512, 2, 4, 4)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_block32 = nn.Sequential(ResBlock(64, 32),
                                     ResBlock(64, 32),
                                     ResBlock(64, 32),
                                     ResBlock(64, 32),
                                     DownBlock(64, 128))

        self.down_block16 = nn.Sequential(ResBlock(128, 32),
                                     ResBlock(128, 32),
                                     ResBlock(128, 32),
                                     ResBlock(128, 32),
                                     #ResBlock(128, 32),
                                     DownBlock(128, 256))
        

        self.block4 = nn.Sequential(ResBlock(256, 64),
                                    ResBlock(256, 64),
                                    ResBlock(256, 64),
                                    ResBlock(256, 64),
                                        Reshaper(),
                                        #ResBlock3D(128, 64),
                                        UpBlock(512, 256)
                                        )
        
        self.up_block8 = nn.Sequential(ResBlock3D(256, 64),
                                        ResBlock3D(256, 64),
                                        UpBlock(256, 128)
                                        )

        self.up_block16 = nn.Sequential(ResBlock3D(128, 32),
                                        #ResBlock3D(64, 64),
                                        UpBlock(128, 64)
                                        )

        self.up_block32 = nn.Sequential(ResBlock3D(64, 32),
                                        #ResBlock3D(32, 32),
                                        UpBlock(64, 32),
                                        )
        self.up_block64 = nn.Sequential(ResBlock3D(32, 32),
                                        #ResBlock3D(16, 16),
                                        )
        self.result = nn.Conv3d(32,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=(0, 1, 1),
                              )

        self.net = nn.Sequential(
                                 self.down_block32,
                                 self.down_block16,
                                 self.block4,
                                 self.up_block8,
                                 self.up_block16,
                                 self.up_block32,
                                 self.up_block64,
                                 self.result,
                                 nn.ReLU(),
                                 )

    def forward(self, x):
        return self.net(x)[:, 0, ...]