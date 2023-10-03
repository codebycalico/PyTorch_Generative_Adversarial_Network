import torch
torch.manual_seed(42)
import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

# Configurations 
# use cuda because we are on GPU
# transfer answers/images to GPU device
# image = image.to(DEVICE)
DEVICE = 'cuda'

# used in trainloader and training loop
BATCH_SIZE = 128

# random noise vector dimension being passed to the generator model
NOISE_DIM = 64

# optimizers parameters
LR = 0.0002
BETA_1 = 0.5
BETA_2 = 0.99

# training variables
# how many times you want to run the training loop
EPOCHS = 20

# Load MNIST Dataset
from torchvision import datasets, transforms as T
train_augs = T.Compose([
    T.RandomRotation((-20, +20)),
    T.ToTensor(), # will convert NumPy or BIL images to Torch Tensor & (h, w, c) -> (c, h, w) which is PyTorch format
])
trainset = datasets.MNIST('MNIST/', download = True, train = True, transform = train_augs)
image, label = trainset[9000]

plt.imshow(image.squeeze(), cmap = 'gray')
print("Total images present in trainset: ", len(trainset))

# Load Dataset into batches
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
print("Total number of batches in trainloader: ", len(trainloader))

dataiter = iter(trainloader)

images, _ = next(dataiter)

print(images.shape)

# 'show_tensor_images' : function is used to plot some of images from the batch
# will plot results from the generator

def show_tensor_images(tensor_img, num_images = 16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()

show_tensor_images(images, num_images = 16)

# Create Discriminator Network
# discriminator network determines if an image is real or fake

from torch import nn
from torchsummary import summary

def get_disc_block(in_channels, out_channels, kernel_size, stride):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2)
  )

# create discriminator network
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.block_1 = get_disc_block(1, 16, (3, 3), 2)
    self.block_2 = get_disc_block(16, 32, (5, 5), 2)
    self.block_3 = get_disc_block(32, 64, (5, 5), 2)

    self.flatten = nn.Flatten()
    self.linear = nn.Linear(in_features = 64, out_features = 1)

  def forward(self, images):
    x1 = self.block_1(images)
    x2 = self.block_2(x1)
    x3 = self.block_3(x2)

    x4 = self.flatten(x3)
    x5 = self.linear(x4)

    return x5
    # not using sigmoid layer because we will be using binary cross entropy with logic loss which takes row outputs
    # which takes the raw output (without any sigmoid) and in the last layer only the sigmoid will be applied

D = Discriminator()
D.to(DEVICE)

summary(D, input_size = (1, 28, 28))

# Create Generator Network
# create the discrimator block
def get_gen_block(in_channels, out_channels, kernel_size, stride, final_block = False):
  if final_block == True:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
        nn.Tanh()
    )

  return nn.Sequential(
      nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
  )

# use the discriminator block generator function to create the network
class Generator(nn.Module):
  def __init__(self, noise_dim):
    super(Generator, self).__init__()

    self.noise_dim = noise_dim
    self.block_1 = get_gen_block(noise_dim, 256, (3, 3), 2)
    self.block_2 = get_gen_block(256, 128, (4, 4), 1)
    self.block_3 = get_gen_block(128, 64, (3, 3), 2)

    self.block_4 = get_gen_block(64, 1, (4, 4), 2, final_block = True)

  def forward(self, random_noise_vec):
    # current shape is (batch size, noise_dim)
    # convert to (batch size, noise_dim, 1, 1) height and width
    x = random_noise_vec.view(-1, self.noise_dim, 1, 1)

    x1 = self.block_1(x)
    x2 = self.block_2(x1)
    x3 = self.block_3(x2)
    x4 = self.block_4(x3) #output from final block

    return x4
  
G = Generator(NOISE_DIM) #create generator network
G.to(DEVICE) #move network to GPU device

# show the summary of the model
summary(G, input_size = (1, NOISE_DIM))

# Replace Random initialized weights to Normal weights

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

D = D.apply(weights_init)
G = G.apply(weights_init)

# Create loss function and load optimizer
def real_loss(disc_pred):
  criterion = nn.BCEWithLogitsLoss()
  ground_truth = torch.ones_like(disc_pred)
  loss = criterion(disc_pred, ground_truth)
  return loss

def fake_loss(disc_pred):
  criterion = nn.BCEWithLogitsLoss()
  ground_truth = torch.zeros_like(disc_pred)
  loss = criterion(disc_pred, ground_truth)
  return loss

# discriminator optimizer
D_opt = torch.optim.Adam(D.parameters(), lr = LR, betas = (BETA_1, BETA_2))
# generator optimizer
G_opt = torch.optim.Adam(G.parameters(), lr = LR, betas = (BETA_1, BETA_2))

# training loop
for i in range(EPOCHS):
  total_d_loss = 0.0 # discriminator loss initially 0
  total_g_loss = 0.0 # generator loss initially 0

  for real_img, _ in tqdm(trainloader):
    real_img.to(DEVICE)
    noise = torch.randn(BATCH_SIZE, NOISE_DIM, device = DEVICE)

    # find loss and update weights for discriminator network

    D_opt.zero_grad()

    fake_img = G(noise)
    D_pred = D(fake_img)
    D_fake_loss = fake_loss(D_pred)

    D_pred = D(fake_img)
    D_real_loss = real_loss(D_pred)

    D_loss = (D_fake_loss + D_real_loss) / 2

    total_d_loss += D_loss.item()

    # find the gradients
    D_loss.backward()
    # find the weight of the network
    D_opt.step()

    #find loss and update weights for generator network

    G_opt.zero_grad()

    noise = torch.randn(BATCH_SIZE, NOISE_DIM, device = DEVICE)

    # fake_img = G(noise) // repetitive? unneeded?
    # D_pred = D(fake_img)
    G_loss = real_loss(D_pred)

    total_g_loss = G_loss.item()

    G_loss.backward()
    G_opt.step()

  avg_d_loss = total_d_loss / len(trainloader)
  avg_g_loss = total_g_loss / len(trainloader)

  print("EPOCH: {} | Discriminator loss: {} | Generator loss: {}".format(i+1, avg_d_loss, avg_g_loss))

  show_tensor_images(fake_img)

# Run after training is completed.
# Now you can use Generator Network to generate handwritten images

noise = torch.randn(BATCH_SIZE, NOISE_DIM, device = DEVICE)
generated_image = G(noise)

show_tensor_images(generated_image)