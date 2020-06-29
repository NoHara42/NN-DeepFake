from __future__ import print_function
#%matplotlib inline
from argparse import ArgumentParser
import random
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from my_models import Generator, Discriminator
from torch.utils.data import DataLoader  # Dataset mangement and for mini batches
import importData # for our own dataset

#Parser for cool parser feeling
parser = ArgumentParser()
parser.add_argument("--epochs",type=int, default=10, help="number of training epochs")
parser.add_argument("--batch_size",type=int, default=64, help="batch_size")

parsed = parser.parse_args()

# Set torch seed and numpy seed manually for reproducibility and for results to be reproducible between CPU and GPU executions
manualSeed = 999
#manualSeed = random.randint(1, 10000) # for new results
#random.seed(manualSeed)

torch.manual_seed(manualSeed)

# Batch size
batch_size = parsed.batch_size
print(batch_size)

# image size
image_size = 64

# hier 3 für RGB, in diesem Falle 1 da in MNIST nur Grau Bilder sind
image_channels = 3

# size of generator input
generator_in = 100

#number of features of generator
generator_features = 64

#number of features of discriminator
discriminator_features = 64

# Number of training epochs
num_epochs = parsed.epochs
print(num_epochs)

# learning rate, defined in paper , https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10 for what it is
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs
number_gpu = 1

#loading of dataset
dataset = importData.OurDataset()
dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and number_gpu > 0) else "cpu")

#weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Create the generator
netG = Generator(number_gpu, image_channels, generator_in, generator_features).to(device)

# Apply the weights_init function to randomly initialize all weights
netG.apply(weights_init)

# Create the Discriminator
netD = Discriminator(number_gpu, image_channels, discriminator_features).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of  vectors for visualization
fixed_noise = torch.randn(64, generator_in, 1, 1, device=device)

#label creation
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        """
        Training of Discriminator:  wants to maximize log(D(x)) + log(1 - D(G(z)))
        """

        #1.Training with real images

        #Set gradient to zero
        netD.zero_grad()

        #Format batch of real images
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        #Forward batch to D
        output = netD(real_cpu).view(-1)

        #Calculate loss on real images batch
        errD_real = criterion(output, label)

        #Calculate gradients for D 
        errD_real.backward()
        D_x = output.mean().item()

        #2.Training with fake images

        # Generate batch of input vectors
        noise = torch.randn(b_size, generator_in, 1, 1, device=device)

        # Generate fake image batch with G
        fake_images = netG(noise)
        label.fill_(fake_label)

        #Label all fake images with D
        output = netD(fake_images.detach()).view(-1)

        # Calculate D's loss on fake batch
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Add the gradients from the real and fake batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()


        """
        Training of Generator: wants to maximize log(D(G(z)))
        """

        #Set gradient to zero
        netG.zero_grad()

        #Set labels to real to confuse D
        label.fill_(real_label)

        # Pass fake images to D after the step
        output = netD(fake_images).view(-1)

        # Calculate G's loss
        errG = criterion(output, label)

        # Calculate gradients for G
        errG.backward()

        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

#Setting up wrtiters to save animations
Writer = animation.writers['ffmpeg']
writer = Writer(fps=3, metadata=dict(artist='Me'), bitrate=1800)

#Animation to see the progress of generated images
matplotlib.rcParams['animation.embed_limit'] = 2**128
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
ani.save('output.mp4', writer = writer)
plt.show
HTML(ani.to_jshtml())

#Plot for G and D Loss
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

