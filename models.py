import torch 
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self, img_channels, d_features):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Anfangsgröße, img_channels(3 weil RGB, bei grau nur 1) x 64p x 64p
            nn.Conv2d( img_channels, d_features, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
            #nach dem ersten filter erhalten wir diese größe : ( https://youtu.be/sTTuaNp5LTw?list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S&t=107) 
            # d_features x 32p x 32p
            nn.Conv2d( d_features, d_features*2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(d_features*2),
            nn.LeakyReLU(0.2),
            # dann diese größe und immer so weiter
            # d_features*2 x 16p x 16p
            nn.Conv2d( d_features * 2, d_features*4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(d_features*4),
            nn.LeakyReLU(0.2),
            # d_features*4 x 8p x 8p
            nn.Conv2d( d_features * 4, d_features*8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(d_features*8),
            nn.LeakyReLU(0.2),
            # d_features*8 x 4p x 4p
            nn.Conv2d( d_features * 8, 1, kernel_size = 4, stride = 2, padding = 0),
            # ----> 1 x 1 x 1, am Ende wird entschieden ob das Bild das verarbeitet wurde, echt war oder nicht
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, noise_channels , img_channels, g_features):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            #genau das gleiche wie Discriminator nur umgekehrt
            # noise_channels x 1 x 1
            nn.ConvTranspose2d(noise_channels, g_features*16, kernel_size = 4, stride = 1, padding = 0),
            nn.BatchNorm2d(g_features*16),
            nn.ReLU(),
            # g_features*16 x 4p x 4p
            nn.ConvTranspose2d(g_features*16, g_features*8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(g_features*8),
            nn.ReLU(),
            # g_features*8 x 8p x 8p
            nn.ConvTranspose2d(g_features*8, g_features*4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(g_features*4),
            nn.ReLU(),
            # g_features*4 x 32p x 32p
            nn.ConvTranspose2d(g_features*4, g_features*2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(g_features*2),
            nn.ReLU(),
            # g_features*2 x 64p x 64p
            nn.ConvTranspose2d(g_features*2, img_channels, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


