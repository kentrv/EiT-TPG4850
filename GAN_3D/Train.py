import torch
from Discriminator import Discriminator
from Generator import Generator
from ShapeNetVoxelizer import Custom3DDataset
from torch.utils.data import DataLoader
from GAN_3D.ShapeNetVoxelizer import ShapeNetVoxelizer
import os
from pandas import DataFrame as df

def model_trainer(dataset_name:str, epochs:int, batch_size:int, input_size:int, lr:float, betas:tuple):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.voxelizer = ShapeNetVoxelizer(resolution=input_size)
        self.dataset_path = os.getcwd()+'/Datasets/ShapeNet/'
        self.dataset_objects = os.listdir(self.dataset_path)
        self.epoch = epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.lr = lr
        self.betas = betas
        self.generator = Generator()
        self.discriminator = Discriminator(input_size)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        self.adversarial_loss = torch.nn.BCELoss()
        
    def train(self):
        for model in self.dataset_objects:
            voxel_array = self.voxelizer.process_obj_file(self.dataset_path+model)
            dataloader = DataLoader(voxel_array, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.epochs):
                for i, data in enumerate(dataloader):
                    valid = torch.ones(data.size(0), 1)
                    fake = torch.zeros(data.size(0), 1)

                    # Generator
                    self.optimizer_G.zero_grad()
                    generated_data = self.generator(data[0])

                    # Discriminator
                    self.optimizer_D.zero_grad()
                    real_loss = self.adversarial_loss(self.discriminator(data), valid)
                    fake_loss = self.adversarial_loss(self.discriminator(generated_data.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    self.optimizer_D.step()

                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")




input_size = 32 # Adjust this based on your data size (32 if 32x32x32, 64 if 64x64x64, etc.)

# Initialize models
generator = Generator()
discriminator = Discriminator(input_size)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = torch.nn.BCELoss()

# DataLoader
dataset = Custom3DDataset(data_paths=['Datasets/datafile'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training Loop
epochs = 50
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        valid = torch.ones(data.size(0), 1)
        fake = torch.zeros(data.size(0), 1)

        # Train Generator
        optimizer_G.zero_grad()
        generated_data = generator(data)
        g_loss = adversarial_loss(discriminator(generated_data), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(data), valid)
        fake_loss = adversarial_loss(discriminator(generated_data.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# Save models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
