import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from dataset import AnimeFaceDataset
from gan_model import Generator, Discriminator
import os
from torchvision import transforms

def main():
    # Hyperparameters
    batch_size = 128
    image_size = 64
    z_dim = 100
    lr = 0.0002
    beta1 = 0.5
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Robust directory setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "data", "animefacedataset", "thumbnails")
    output_dir = os.path.join(BASE_DIR, "outputs")
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Print the directories to debug any path issues
    print("Looking for images in:", data_dir)
    print("Saving outputs to:", output_dir)
    print("Saving models to:", model_dir)

    # Dataset and Dataloader
    dataset = AnimeFaceDataset(data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Models
    netG = Generator(z_dim).to(device)
    netD = Discriminator().to(device)

    # Loss and Optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Labels
    real_label = 1.
    fake_label = 0.

    # Training Loop
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # 1. Train Discriminator
            ############################
            netD.zero_grad()
            real_images = data.to(device)
            b_size = real_images.size(0)
            labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_images).view(-1)
            lossD_real = criterion(output, labels)
            lossD_real.backward()

            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_images = netG(noise)
            labels.fill_(fake_label)

            output = netD(fake_images.detach()).view(-1)
            lossD_fake = criterion(output, labels)
            lossD_fake.backward()

            lossD = lossD_real + lossD_fake
            optimizerD.step()

            ############################
            # 2. Train Generator
            ############################
            netG.zero_grad()
            labels.fill_(real_label)  # Generator tries to fool discriminator
            output = netD(fake_images).view(-1)
            lossG = criterion(output, labels)
            lossG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch+1}/{epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

        # Save sample images and models
        vutils.save_image(fake_images.detach()[:64],
                          os.path.join(output_dir, f"epoch_{epoch+1}.png"),
                          normalize=True)

        torch.save(netG.state_dict(), os.path.join(model_dir, f"netG_epoch_{epoch+1}.pth"))
        torch.save(netD.state_dict(), os.path.join(model_dir, f"netD_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()