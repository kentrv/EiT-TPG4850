import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def add_random_corruption(voxels, corruption_rate=0.3):
    """
    Corrupts a portion of the voxel data by setting a fraction of voxels to 0.
    
    Parameters:
    - voxels: A PyTorch tensor of shape (batch_size, channels, depth, height, width),
              representing the voxel data.
    - corruption_rate: A float representing the fraction of voxels to corrupt.
    
    Returns:
    - A new PyTorch tensor with the same shape as `voxels`, where a portion of the
      voxels has been set to 0.
    """
    # Ensure corruption rate is between 0 and 1
    corruption_rate = max(0, min(corruption_rate, 1))
    
    # Create a mask with the same shape as `voxels` where some values are set to 0
    # based on the corruption rate
    mask = torch.rand_like(voxels) > corruption_rate
    
    # Apply the mask to `voxels`, setting some voxels to 0
    corrupted_voxels = voxels * mask
    
    return corrupted_voxels



class PhaseOneTrainer:
    def __init__(self, generator, discriminator, dataset, batch_size, lr_g_initial=1e-5, lr_g_jointly=1e-4, lr_d_initial=1e-6, betas=(0.5, 0.999), alpha1=0.001, alpha2=0.999):
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.batch_size = batch_size
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_g_initial, betas=betas)
        self.optimizer_G_jointly = torch.optim.Adam(self.generator.parameters(), lr=lr_g_jointly, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d_initial, betas=betas)
        self.adversarial_loss = torch.nn.BCELoss()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def L_3D_ED_GAN(self, generated_voxels, real_voxels, output_discriminator):
        """Calculate the combined L_3D_ED_GAN loss."""
        loss_GAN = self.adversarial_loss(output_discriminator, torch.ones_like(output_discriminator))
        loss_recon = PhaseOneTrainer.reconstruction_loss(generated_voxels, real_voxels)
        return self.alpha1 * loss_GAN + self.alpha2 * loss_recon

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

        for epoch in epochs:
            for i, (voxels, _) in enumerate(dataloader):
                valid = torch.ones((voxels.size(0), 1), device=voxels.device)
                fake = torch.zeros((voxels.size(0), 1), device=voxels.device)

                corrupted_voxels = add_random_corruption(voxels)
                # Generator forward pass with corrupted voxels
                self.optimizer_G_jointly.zero_grad()
                generated_voxels = self.generator(corrupted_voxels)
                output_discriminator = self.discriminator(generated_voxels)

                # Combined GAN and Reconstruction Loss for Generator
                combined_loss = self.L_3D_ED_GAN(generated_voxels, voxels, output_discriminator)
                combined_loss.backward()
                self.optimizer_G_jointly.step()

                # Discriminator forward pass
                self.optimizer_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(voxels), valid)
                fake_loss = self.adversarial_loss(output_discriminator.detach(), fake)
                d_loss = (real_loss + fake_loss) / 2

                if d_loss.item() < discriminator_update_threshold:
                    d_loss.backward()
                    self.optimizer_D.step()

                if i % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i}, D Loss: {d_loss.item()}, G Loss: {combined_loss.item()}")

                    
    @staticmethod
    def reconstruction_loss(output, target):
        """Reconstruction loss based on binary cross-entropy."""
        return F.binary_cross_entropy(output, target)
    

class PhaseTwoTrainer:
    def __init__(self, lrcn, dataset, batch_size, lr_initial=1e-4, betas=(0.5, 0.999)):
        self.lrcn = lrcn
        self.dataset = dataset
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.lrcn.parameters(), lr=lr_initial, betas=betas)
        self.criterion = torch.nn.BCELoss()

    def train(self, epochs=20):
        """Training the LRCN model."""
        self.lrcn.train()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            for i, (slices, _) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.lrcn(slices)
                loss = self.criterion(output, slices)
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
                    
    def criterion(self, output, target):
        """Reconstruction loss based on binary cross-entropy."""
        return F.binary_cross_entropy(output, target)
