import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.unet import UNet
from config import *
from utils.data_utils import apply_blur, apply_noise

def load_model(model_path):
    """Load the trained model"""
    model = UNet(n_channels=IN_CHANNELS, n_classes=OUT_CHANNELS)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])  # Extract model state dict from checkpoint
    model.to(DEVICE)
    model.eval()
    return model

def process_image(image_path, model):
    """Process a single image through the model"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Apply degradation
    blurred = apply_blur(image_tensor, BLUR_KERNEL_SIZE, BLUR_SIGMA)
    degraded = apply_noise(blurred, NOISE_STD)
    
    # Restore image
    with torch.no_grad():
        restored = model(degraded)
    
    # Convert tensors to numpy arrays for visualization
    def tensor_to_numpy(tensor):
        tensor = tensor.squeeze(0).cpu()
        tensor = tensor * torch.tensor(NORMALIZE_STD).view(3, 1, 1)
        tensor = tensor + torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy()
    
    original = tensor_to_numpy(image_tensor)
    degraded = tensor_to_numpy(degraded)
    restored = tensor_to_numpy(restored)
    
    return original, degraded, restored

def visualize_results(original, degraded, restored, save_path=None):
    """Visualize and optionally save the results"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(degraded)
    plt.title('Degraded Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(restored)
    plt.title('Restored Image')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load the best model
    model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first.")
        return
    
    model = load_model(model_path)
    print("Model loaded successfully.")
    
    # Get test images
    test_images = [os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR) 
                  if f.endswith(('.png', '.jpg', '.jpeg'))][:3]  # Test on first 3 images
    
    # Process and visualize results
    for i, image_path in enumerate(test_images):
        print(f"\nProcessing image {i+1}: {os.path.basename(image_path)}")
        original, degraded, restored = process_image(image_path, model)
        save_path = os.path.join('results', f'result_{i+1}.png')
        visualize_results(original, degraded, restored, save_path)

if __name__ == '__main__':
    main() 