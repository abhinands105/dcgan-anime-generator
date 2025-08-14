import torch
import torchvision.utils as vutils
from gan_model import Generator
import os

# Configs
z_dim = 100
num_images = 64

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Correct paths
model_path = os.path.join(script_dir, "models", "netG_epoch_50.pth")
output_path = os.path.join(script_dir, "outputs", "generated.png")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Generator
netG = Generator(z_dim).to(device)
netG.load_state_dict(torch.load(model_path, map_location=device))
netG.eval()

# Generate random noise
noise = torch.randn(num_images, z_dim, 1, 1, device=device)

# Generate fake images
with torch.no_grad():
    fake_images = netG(noise).detach().cpu()

# Save to output file
os.makedirs(os.path.dirname(output_path), exist_ok=True)
vutils.save_image(fake_images, output_path, nrow=8, normalize=True)

print(f"✅ Generated image grid saved at: {output_path}")