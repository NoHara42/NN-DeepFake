import torch
import torchvision
import torch.nn as nn # Neural Networks functions
import torch.optim as optim # Optimization algos
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  # Dataset mangement and for mini batches
from torch.utils.tensorboard import SummaryWriter # for nice visu
from models import Generator, Discriminator



#Hyperparameters

# learning rate, defined in paper
lr = 0.0002

#work batches
batch_size = 64

#image size important for Conv2dLayers
image_size = 64

channels_img = 1

#length of noise vector
channels_noise = 256

num_epochs = 10

g_features = 16
d_features = 16

my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    ])


dataset = datasets.MNIST(root='dataset/', train = True, transform = my_transforms, download = True)
dataloader = DataLoader(dataset,batch_size = batch_size, shuffle = True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Create Discriminator and Generator

netD = Discriminator(channels_img, d_features).to(device)
netG = Generator(channels_noise, channels_img, g_features).to(device)


#Setup Optimizer for G and D
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas=(0.5, 0.999))

netD.train()
netG.train()

#print(netD)
#print(netG)



#loss function from paper
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

#to see progession in image generation
fixed_noise  = torch.randn(64, channels_noise, 1, 1,).to(device)

writer_real = SummaryWriter(f'runs/DCGAN_MNIST/test_real')
writer_fake = SummaryWriter(f'runs/DCGAN_MNIST/test_fake')
print("Starting training....")


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


        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                Loss D: {lossD:.4f}, Loss G: {lossG:.4f} D(x): {D_x:.4f} ')

            with torch.no_grad():
                fakeImages = netG(fixed_noise)

                img_grid_real = torchvision.utils.make_grid(data[:32] , normalize = True)
                img_grid_fake = torchvision.utils.make_grid(fakeImages[:32] , normalize = True)
                writer_real.add_image('MNIST Real Images', img_grid_real)
                writer_fake.add_image('MNIST Fake Images', img_grid_fake)