import torch
import torchvision
import torch.nn as nn # Neural Networks functions
import torch.optim as optim # Optimization algos
import torchvision.datasets as datasets #das Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  # Dataset mangement and for mini batches
from torch.utils.tensorboard import SummaryWriter # für die Visualization
from models import Generator, Discriminator # die definierten Modelle
import matplotlib.pyplot as plt
import importData
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
import torchvision.utils as vutils
import time



#Hyperparameters

# learning rate, defined in paper , https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10 for what it is
lr = 0.0002

#work batches
batch_size = 64

#image size important for Conv2dLayers, die Bilder haben zwar 28x28 werden aber duch my_transforms resized
image_size = 64

# hier 3 für RGB, in diesem Falle 1 da in MNIST nur Grau Bilder sind
channels_img = 3

#length of noise vector, kann varieren
channels_noise = 256

#Anzahl der Durchläufe
num_epochs = 10

#Anzahl der Features die aufgenommen werden, je mehr desto besser, dauert dann aber auch länger
g_features = 16
d_features = 16

#Wenn GPU vorhanden die Cuda aktiviert hat, wird diese als Gerät benutzt sonst die CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Compose ist zum Aneinaderreihen von Bild Veränderungen
my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    #konvertiert Bild mit Koordinaten in einen FloatTensor ( siehe https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor)
    transforms.ToTensor(),
    # Normailzes the tensor but does not mutate it
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

#Dataset wird geladen
# dataset = datasets.MNIST(root='dataset/', train = True, transform = my_transforms, download = True)
dataset = importData.OurDataset()
dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = True)

#Create Discriminator and Generator
netD = Discriminator(channels_img, d_features).to(device)
netG = Generator(channels_noise, channels_img, g_features).to(device)


#Setup Optimizer for G and D
# https://www.youtube.com/watch?v=mdKjMPmcWjY und https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c als Ressourcen für Adamne
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas=(0.5, 0.999))

#send them into train mode
netD.train()
netG.train()

#print(netD)
#print(netG)



#loss function from paper
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

#um forschritt der bilder zu sehen setzen wir die noise fixed und nehmen nicht immer eine andere
fixed_noise  = torch.randn(64, channels_noise, 1, 1,).to(device)

#erstellt die dirs für tensorboard das dann mit "tensorboard --logdir runs" aufgerufen werden kann
writer_real = SummaryWriter(f'runs/DCGAN_JESTER/test_real')
writer_fake = SummaryWriter(f'runs/DCGAN_JESTER/test_fake')

#lists for plotting
img_list = []
G_losses = []
D_losses = []

print("Starting training....")

#hier bin ich auch noch nicht komplett durchgestiegen
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device)
        batch_size = data.shape[0]

        # Train discriminator, we want to maximize log(D(x)) + log(1- D(G(z)))
            # 1. send in all real images
            # 2. send in all fake images

        netD.zero_grad()
        # all real images
        label = (torch.ones(batch_size) *0.9).to(device)

        # between 0 and 1, we want them to be 1
        output = netD(data).reshape(-1)

        lossD_real = criterion(output, label)

        # compute mean confidence
        D_x = output.mean().item()

        noise = torch.randn(batch_size, channels_noise,1,1).to(device)
        fakeImages = netG(noise)
        label = (torch.ones(batch_size)* 0.1).to(device)

        output = netD(fakeImages.detach()).reshape(-1)
        lossD_fake = criterion(output, label)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()



        # Train generator, we want to maximize log(D(G(z)))
        netG.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = netD(fakeImages).reshape(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        #das ist einfach nur Print um zu sehen wie weit man ist und um Bilder für tensorboard zu generieren
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                Loss D: {lossD:.4f}, Loss G: {lossG:.4f} D(x): {D_x:.4f} ')

            # Save Losses for plotting later
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            with torch.no_grad():
                fakeImages = netG(fixed_noise).detach().cpu()

                img_grid_real = torchvision.utils.make_grid(data[:32] , normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fakeImages[:32] , normalize = True)
                writer_real.add_image('MNIST Real Images', img_grid_real)
                writer_fake.add_image('MNIST Fake Images', img_grid_fake)

                #imgs for plots
                img_list.append(torchvision.utils.make_grid(fakeImages, padding=2, normalize=True))



#plot of D and G losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Animation
'''
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
'''

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()

# Plot the fake images from the last epoch
plt.figure(figsize=(15,15))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()


