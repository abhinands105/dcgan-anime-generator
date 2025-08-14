# dcgan-anime-generator 

Generate anime faces using DCGAN and PyTorch

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate 64×64 anime-style faces.  
It uses **PyTorch** for model definition and training, with:
- **Generator**: Upsamples random noise into realistic anime faces.
- **Discriminator**: Distinguishes real images from generated ones.
- **Training**: Alternates between updating the Discriminator and Generator using Binary Cross-Entropy loss.
- **Output**: Saves generated sample grids per epoch and trained model checkpoints for later use.

The workflow:
1. Load anime face dataset.
2. Train DCGAN for ~50 epochs.
3. Save model weights + generated images.
4. Use the trained Generator to create new anime faces from random noise.
