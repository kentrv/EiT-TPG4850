import torch
from Discriminator import Discriminator
from Generator import Generator
from ShapeNetVoxelizer import Custom3DDataset
from torch.utils.data import DataLoader

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
