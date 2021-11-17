import kornia
import torch
import torch.nn as nn


class CARE_projection(nn.Module):
    """
    Network architecture from Weigert et al., Nature Methods, 2018, just the projection part.
    """

    def __init__(self, n_filter=8):
        super().__init__()
        # encode
        self.conv1 = self.conv(1, n_filter)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=(0, 0, 0))
        self.conv2 = self.conv(n_filter, n_filter)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=(0, 0, 0))
        self.conv3 = self.conv(n_filter, n_filter)
        self.up1 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
        self.conv4 = self.conv(n_filter, n_filter)
        self.up2 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear', align_corners=False)
        self.conv5 = self.conv(n_filter, n_filter)
        self.out = self.final(n_filter, 1)

    def conv(self, in_channels, out_channels, dropout=0.0):
        block = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 5, 5), in_channels=in_channels, out_channels=out_channels, padding=(1, 2, 2)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(dropout)
        )
        return block

    def sobel(self, x):
        shape = x.shape
        x = x.view((shape[0], *shape[-3:]))
        x = kornia.sobel(x)
        return x

    def final(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 5), padding=(2, 2, 2)),
            nn.Sigmoid(),
        )
        return block

    def forward(self, x):
        c1 = self.conv1(x)
        m1 = self.maxpool1(c1)
        c2 = self.conv2(m1)
        m2 = self.maxpool2(c2)
        c3 = self.conv3(m2)
        u1 = self.up1(c3)
        c4 = self.conv4(u1)
        u2 = self.up2(c4)
        c5 = self.conv5(u2)
        mask = self.out(c5)
        mult = torch.mul(mask, x)
        out = torch.max(mult, 2)[0]
        edge = self.sobel(mask)
        return out, mask, edge
