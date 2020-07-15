from __future__ import print_function
#%matplotlib inline
from argparse import ArgumentParser
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from my_models import Generator, Discriminator
from torch.utils.data import DataLoader  # Dataset mangement and for mini batches
import importData # for our own dataset
import import_lol_dataset
import time
from datetime import datetime, timedelta
import random

#for measuring time
start = time.time()
def formatSeconds(sekunden):
    sec = timedelta(seconds=int(sekunden))
    d = datetime(1,1,1) + sec
    #print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))


#Parser for cool parser feeling
parser = ArgumentParser()
parser.add_argument("--epochs",type=int, default=10, help="number of training epochs")
parser.add_argument("--batch_size",type=int, default=64, help="batch_size")
parser.add_argument("--dataset", type=str, default = "jester", help = "either jester or lol")
parser.add_argument('--netG', default='', help="path to netG checkpoint (to continue training)")
parser.add_argument('--netD', default='', help="path to netD checkpoint (to continue training)")
parser.add_argument('--outf', default='output', help='folder to save model checkpoints')
parser.add_argument('--saveLoss', default='', help='folder containing G and D loss list to continue plotting')
parser.add_argument('--save',type = bool, default= False, help='if saving the loosses and checkpoints is wanted')

parsed = parser.parse_args()

try:
    os.makedirs(parsed.outf)
except OSError:
    pass

# Set torch seed manually for reproducibility and for results to be reproducible between CPU and GPU executions
manualSeed = 42
#manualSeed = random.randint(1, 10000) # for new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#to improve cuda speed
cudnn.benchmark = True

# Batch size
batch_size = parsed.batch_size

# image size
image_size = 64

#3 for RGB
image_channels = 3

# size of generator input
generator_in = 100

#number of features of generator
generator_features = 64

#number of features of discriminator
discriminator_features = 64

# Number of training epochs
num_epochs = parsed.epochs

# learning rate, defined in paper , https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10 for what it is
#lr = 0.0002
#lr test for lol dataset
lr = 0.0003

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs
number_gpu = 1

#which dataset is gonna be used
dataset_name = parsed.dataset

#loading of dataset
if dataset_name == "jester":
    dataset = importData.OurDataset()
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
elif dataset_name == "lol":
    dataset = import_lol_dataset.OurDataset()
    dataloader = DataLoader(dataset ,batch_size = batch_size, shuffle = True)
else: 
    print("Please choose either the jester or lol dataset")
    quit()

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

#load checkpoint for G if one is available
if parsed.netG != '':
    netG.load_state_dict(torch.load(parsed.netG))
    print("Checkpoint for G has been loaded")
#print(netG)

# Create the Discriminator
netD = Discriminator(number_gpu, image_channels, discriminator_features).to(device)


# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

#load checkpoint for D if one is available
if parsed.netD != '':
    netD.load_state_dict(torch.load(parsed.netD))
    print("Checkpoint for D has been loaded")
#print(netD)

# Initialize BCELoss function
loss_function = nn.BCELoss()

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

print("Starting training")
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

        #Transfer data to CPU/GPU
        real_data = data[0].to(device)
        batch_size = real_data.size(0)

        #Label smoothing test
        label_smoothing_real = random.uniform(0.7, 1.2)

        #label = torch.full((batch_size,), label_smoothing_real, device=device)
        label = torch.full((batch_size,), real_label, device=device)

        #Forward batch to D
        output = netD(real_data).view(-1)


        #Calculate loss on real images batch
        errD_real = loss_function(output, label)
        

        #Calculate gradients for D 
        errD_real.backward()

        D_x = output.mean().item()


        #2.Training with fake images

        # Generate batch of input vectors
        #Sampled from a normal distribution
        noise = torch.randn(batch_size, generator_in, 1, 1, device=device)

        # Generate fake image batch with G
        fake_images = netG(noise)

        #label smoothing 
        label_smoothing_fake = random.uniform(0.0, 0.3)

        label.fill_(label_smoothing_fake)

        #Label all fake images with D
        output = netD(fake_images.detach()).view(-1)

        # Calculate D's loss on fake batch
        errD_fake = loss_function(output, label)
        
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Add the gradients from the real and fake batches
        errD = errD_real + errD_fake
        print(errD)
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
        errG = loss_function(output, label)

        # Calculate gradients for G
        errG.backward()

        D_G_z = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if dataset_name == 'lol' :
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                    % (epoch+1, num_epochs, i +1, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z))

            #for time measuring
            timeTillNow = time.time()
            elapsedTime = timeTillNow - start
            formatSeconds(elapsedTime)

        else:
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                    % (epoch+1, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z))

                #for time measuring
                timeTillNow = time.time()
                elapsedTime = timeTillNow - start
                formatSeconds(elapsedTime)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if dataset_name == 'lol' :
            if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        else:
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


if parsed.netD != '':
    num_epochs = int(parsed.netD.split("_")[2].split(".")[0]) + num_epochs

if parsed.saveLoss != "":
    G_losses_last_run = list(np.load(os.path.join(parsed.saveLoss, "G_losses.npy")))
    D_losses_last_run = list(np.load(os.path.join(parsed.saveLoss, "D_losses.npy")))

    D_losses = D_losses_last_run + D_losses
    G_losses = G_losses_last_run + G_losses

#save checkpoints for G and D, as well as the Loss lists
if parsed.save:
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (parsed.outf, num_epochs))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (parsed.outf, num_epochs))

    save_D_loss = os.path.join(parsed.outf, "D_losses.npy")
    save_G_loss = os.path.join(parsed.outf, "G_losses.npy")

    np.save(save_G_loss, np.array(G_losses))
    np.save(save_D_loss, np.array(D_losses))

#Setting up wrtiters to save animations
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

#Animation to see the progress of generated images
matplotlib.rcParams['animation.embed_limit'] = 2**128
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
animation = "animation_after_" + str(num_epochs) + "_epochs.mp4"
save_animation = os.path.join(parsed.outf, animation)
ani.save(save_animation, writer = writer)
plt.close()
HTML(ani.to_jshtml())

#Plot for G and D Loss
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
G_and_D_loss = "Loss_after_" + str(num_epochs) + "_epochs.png"
save_G_and_D_loss = os.path.join(parsed.outf, G_and_D_loss)
plt.savefig(save_G_and_D_loss)
plt.close()

"""
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))


# Plot the real images
plt.figure(figsize=(20,20))
plt.subplot(1,1,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()
"""


# Plot the fake images from the last epoch
plt.figure(figsize=(20,20))
plt.subplot(1,1,1)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
images_last_epoch = "fakeImages_after_" + str(num_epochs) + "_epochs.png"
save_Images = os.path.join(parsed.outf, images_last_epoch)
plt.savefig(save_Images)
plt.close()


print(iters)

#for time measuring
end = time.time()
totalTime = end - start
print("Total Runtime: ")
formatSeconds(totalTime)
