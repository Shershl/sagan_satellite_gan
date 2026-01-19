# Self-Attention GAN for Satellite Image Synthesis

This project implements a class-conditional Self-Attention GAN (SAGAN) for satellite image synthesis using PyTorch.
It was developed as a bachelor thesis project and focuses on stable GAN training and evaluation.

## Key Features
- Class-conditional GAN architecture
- Self-Attention modules in Generator and Discriminator
- Spectral Normalization for training stability
- Hinge loss formulation
- Mixed Precision Training (AMP)
- Custom PyTorch Dataset and DataLoader
- Model evaluation using F.plugin FID and LPIPS
- Visualization of generated samples, training losses, and attention heatmaps
- Checkpointing and experiment logging

## Project Structure
- `generator.py` – Generator architecture with self-attention blocks  
- `discriminator.py` – Discriminator with spectral normalization  
- `self_attention.py` – Self-attention module  
- `dataset_loader.py` – Custom PyTorch dataset  
- `Train.py` – Training pipeline  
- `evaluate_metrics.py` – FID and LPIPS evaluation  
- `plot_losses.py` – Training loss visualization  

## Technologies
Python, PyTorch, Torchvision, NumPy, GANs, Computer Vision

## Notes
The dataset is not included in the repository.
