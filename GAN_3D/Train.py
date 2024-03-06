import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class PhaseOneTrainer:
    def __init__(self, generator, discriminator, dataset, batch_size, lr_g_initial=1e-5, lr_d_initial=1e-6, betas=(0.5, 0.999), alpha1=0.001, alpha2=0.999):
        self.generator = generator
        self.discriminator = discriminator
        self.dataset = dataset
        self.batch_size = batch_size
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_g_initial, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d_initial, betas=betas)
        self.adversarial_loss = torch.nn.BCELoss()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def L_3D_ED_GAN(self, generated_voxels, real_voxels, output_discriminator):
        """Calculate the combined L_3D_ED_GAN loss."""
        loss_GAN = self.adversarial_loss(output_discriminator, torch.ones_like(output_discriminator))
        loss_recon = PhaseOneTrainer.reconstruction_loss(generated_voxels, real_voxels)
        return self.alpha1 * loss_GAN + self.alpha2 * loss_recon

    @staticmethod
    def reconstruction_loss(output, target):
        """Reconstruction loss based on binary cross-entropy."""
        return F.binary_cross_entropy(output, target)

    def train_jointly(self, epochs=100, discriminator_update_threshold=0.8):
        """Joint training of Generator and Discriminator."""
        self.generator.train()
        self.discriminator.train()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
        for epoch in range(epochs):
            for i, (voxels, _) in enumerate(dataloader):
                valid = torch.ones((voxels.size(0), 1), device=voxels.device)
                fake = torch.zeros((voxels.size(0), 1), device=voxels.device)
    
                # Corrupt voxels here if needed
                # Assuming voxels are already corrupted as input for this phase
    
                # Generator forward pass
                self.optimizer_G.zero_grad()
                generated_voxels = self.generator(voxels)
                output_discriminator = self.discriminator(generated_voxels)
                loss_GAN = self.adversarial_loss(output_discriminator, valid)
                
                # Combined GAN and Reconstruction Loss for Generator
                combined_loss = self.L_3D_ED_GAN(generated_voxels, voxels, output_discriminator)
                combined_loss.backward()
                self.optimizer_G.step()
    
                # Discriminator forward pass
                self.optimizer_D.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(voxels), valid)
                fake_loss = self.adversarial_loss(output_discriminator.detach(), fake)
                d_loss = (real_loss + fake_loss) / 2
    
                if d_loss.item() < discriminator_update_threshold:
                    d_loss.backward()
                    self.optimizer_D.step()
    
                if i % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i}, D Loss: {d_loss.item()}, G Loss: loss_GAN.item()")  # Adjust as needed
    