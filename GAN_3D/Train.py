import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

class PhaseOneTrainer:
    def __init__(self, generator, discriminator, dataset, batch_size, lr_g_initial=1e-5, lr_d_initial=1e-6, betas=(0.5, 0.999)):
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.batch_size = batch_size
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_g_initial, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d_initial, betas=betas)
        self.adversarial_loss = torch.nn.BCELoss()

    def train_generator_only(self, epochs=20):
        """Generator training with reconstruction loss."""
        self.generator.train()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            for i, (voxels, _) in enumerate(dataloader):
                self.optimizer_G.zero_grad()

                generated_voxels = self.generator(voxels)
                loss = F.binary_cross_entropy(generated_voxels, voxels)  # Using BCE as reconstruction loss here
                loss.backward()
                self.optimizer_G.step()

                if i % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

    def train_jointly(self, epochs=100, discriminator_update_threshold=0.8):
        """Joint training of Generator and Discriminator."""
        self.generator.train()
        self.discriminator.train()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            for i, (voxels, _) in enumerate(dataloader):
                # Prepare labels for real (1s) and fake (0s) samples
                valid = torch.ones((voxels.size(0), 1), device=voxels.device)
                fake = torch.zeros((voxels.size(0), 1), device=voxels.device)

                # Generator forward pass
                self.optimizer_G.zero_grad()
                generated_voxels = self.generator(voxels)
                g_loss = self.adversarial_loss(self.discriminator(generated_voxels), valid)
                g_loss.backward()
                self.optimizer_G.step()

                # Discriminator forward pass
                self.optimizer_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(voxels), valid)
                fake_loss = self.adversarial_loss(self.discriminator(generated_voxels.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                if d_loss.item() < discriminator_update_threshold:
                    d_loss.backward()
                    self.optimizer_D.step()

                if i % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    @staticmethod
    def reconstruction_loss(output, target):
        """Reconstruction loss based on binary cross-entropy."""
        return F.binary_cross_entropy(output, target)
