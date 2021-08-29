# This will contain the Segmentation Model structre

import torch.nn as nn
import torch
import torch.nn.functional as F


# The Below implementation is based on 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation Research Paper

class conv_block(nn.Module):
    def __init__(self, in_channels : int , out_channels: int):
        super(conv_block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
          )
    def forward(self,x):
        return self.conv_block(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels:int):
        super(DownBlock, self).__init__()
        self.down_layer = nn.Sequential(
            nn.MaxPool3d(2, 2),
            conv_block(in_channels, out_channels)
        )
    def forward(self, x):
        return self.down_layer(x)


# Took referance # 
class UpSampleblock(nn.Module):
    def __init__(self, in_channels : int, out_channels:int, trilinear : bool=True):
        super(UpSampleblock, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super(UNet3d, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        #Encoding Layers 
        self.conv = conv_block(in_channels, n_channels)
        self.d1 = DownBlock(n_channels, 2 * n_channels)
        self.d2 = DownBlock(2 * n_channels, 4 * n_channels)
        self.d3 = DownBlock(4 * n_channels, 8 * n_channels)
        self.d4 = DownBlock(8 * n_channels, 8 * n_channels)
        # Decoding Layers
        self.up1 = UpSampleblock(16 * n_channels, 4 * n_channels)
        self.up2 = UpSampleblock(8 * n_channels, 2 * n_channels)
        self.up3 = UpSampleblock(4 * n_channels, n_channels)
        self.up4 = UpSampleblock(2 * n_channels, n_channels)
        # Output Layer
        self.out  = nn.Conv3d(n_channels, n_classes, kernel_size = 1)

    def forward(self, x):
        # Encoder
        x1 = self.conv(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        # Decoder
        mask = self.up1(x5, x4)
        mask = self.up2(mask, x3)
        mask = self.up3(mask, x2)
        mask = self.up4(mask, x1)
        # Final Output 
        mask = self.out(mask)
        return mask


if __name__ == '__main__':
	model = UNet3d(in_channels=1, n_classes=2, n_channels = 6)
	x = torch.randn(1,1, 78, 120, 120)
	res = model(x)
	print(res.shape)