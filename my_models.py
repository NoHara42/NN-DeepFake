import torch 
import torch.nn as nn


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu, image_channels, generator_in, g_features):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is generator_in going into a convolution
            nn.ConvTranspose2d( generator_in, g_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_features * 8),
            nn.ReLU(True),
            # state size. (g_features*8) x 4 x 4
            nn.ConvTranspose2d(g_features * 8, g_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features * 4),
            nn.ReLU(True),
            # state size. (g_features*4) x 8 x 8
            nn.ConvTranspose2d( g_features * 4, g_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features * 2),
            nn.ReLU(True),
            # state size. (g_features*2) x 16 x 16
            nn.ConvTranspose2d( g_features * 2, g_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features),
            nn.ReLU(True),
            # state size. (g_features) x 32 x 32
            nn.ConvTranspose2d( g_features, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (image_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, image_channels, d_features):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (image_channels) x 64 x 64
            nn.Conv2d(image_channels, d_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d_features) x 32 x 32
            nn.Conv2d(d_features, d_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d_features*2) x 16 x 16
            nn.Conv2d(d_features * 2, d_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d_features*4) x 8 x 8
            nn.Conv2d(d_features * 4, d_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d_features*8) x 4 x 4
            nn.Conv2d(d_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)