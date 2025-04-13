import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from config import *
from utils.data_utils import get_dataloaders
from utils.metrics import evaluate_batch
from models.unet import get_model as get_unet_model
from models.dncnn import get_model as get_dncnn_model

def load_checkpoint(model, optimizer, checkpoint_dir):
    """Load the latest checkpoint if it exists."""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return 0  # Start from epoch 0 if no checkpoints exist
    
    # Try to find the best model checkpoint first
    if 'best_model.pth' in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        print("Loading best model checkpoint")
    else:
        # Otherwise, find the latest epoch checkpoint
        try:
            # Try to get checkpoints with epoch numbers
            epoch_checkpoints = [f for f in checkpoints if f.startswith('checkpoint_epoch_')]
            if epoch_checkpoints:
                latest_checkpoint = max(epoch_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                print(f"Loading latest epoch checkpoint: {latest_checkpoint}")
            else:
                # If no epoch checkpoints, use the first .pth file found
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                print(f"Loading checkpoint: {checkpoints[0]}")
        except (ValueError, IndexError):
            # If there's any error in parsing the filename, use the first .pth file
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
            print(f"Loading checkpoint: {checkpoints[0]}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    return start_epoch

def train():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if MODEL_TYPE == "unet":
        model = get_unet_model()
    else:
        model = get_dncnn_model()
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders()
    
    # Create directories for checkpoints and logs
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_DIR)
    print(f"Starting training from epoch {start_epoch}")
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {'psnr': [], 'ssim': []}
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]') as pbar:
            for degraded, clean in pbar:
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(degraded)
                loss = criterion(output, clean)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                metrics = evaluate_batch(output, clean)
                train_metrics['psnr'].append(metrics['psnr'])
                train_metrics['ssim'].append(metrics['ssim'])
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        avg_train_psnr = np.mean(train_metrics['psnr'])
        avg_train_ssim = np.mean(train_metrics['ssim'])
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'psnr': [], 'ssim': []}
        
        with torch.no_grad():
            for degraded, clean in tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]'):
                degraded = degraded.to(device)
                clean = clean.to(device)
                
                # Forward pass
                output = model(degraded)
                loss = criterion(output, clean)
                
                # Update metrics
                val_loss += loss.item()
                metrics = evaluate_batch(output, clean)
                val_metrics['psnr'].append(metrics['psnr'])
                val_metrics['ssim'].append(metrics['ssim'])
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        avg_val_psnr = np.mean(val_metrics['psnr'])
        avg_val_ssim = np.mean(val_metrics['ssim'])
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('PSNR/train', avg_train_psnr, epoch)
        writer.add_scalar('PSNR/val', avg_val_psnr, epoch)
        writer.add_scalar('SSIM/train', avg_train_ssim, epoch)
        writer.add_scalar('SSIM/val', avg_val_ssim, epoch)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, PSNR: {avg_train_psnr:.2f}, SSIM: {avg_train_ssim:.4f}')
        print(f'Val Loss: {val_loss:.4f}, PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
        
        # Save checkpoint periodically
        if (epoch + 1) % SAVE_FREQUENCY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    train() 