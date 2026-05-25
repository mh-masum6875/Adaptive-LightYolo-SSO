"""
Training Script for Early Exit WSN-YOLO
=======================================
Paper 3: Adaptive and Energy-Efficient Anomaly Detection in WSNs

Place this file in your project ROOT directory.

Usage:
    python train_early_exit.py --cfg cfg/training/yolo_wsn_sso_optimized.yaml --data data/wsn_dataset.yaml --epochs 100

Author: Maidul Hasan Masum
University of Malaya
"""

import argparse
import logging
import os
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2

# Setup path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import model
from models.yolo import EarlyExitModel
from models.early_exit import EarlyExitLoss, ExitStatistics

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class WSNDataset(Dataset):
    """WSN Anomaly Detection Dataset - YOLO Format."""
    
    def __init__(self, img_dir, label_dir=None, img_size=256, augment=False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.img_size = img_size
        self.augment = augment
        
        # Find images
        self.img_files = sorted(
            list(self.img_dir.glob('*.png')) + 
            list(self.img_dir.glob('*.jpg')) +
            list(self.img_dir.glob('*.jpeg'))
        )
        
        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        logger.info(f"Loaded {len(self.img_files)} images from {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        
        # Get label
        label = self._get_label(img_path)
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label
    
    def _get_label(self, img_path):
        """Get label from file or filename."""
        # Try label file (YOLO format)
        if self.label_dir and self.label_dir.exists():
            label_path = self.label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            # YOLO format: class_id x_center y_center width height
                            return int(float(line.split()[0]))
                except:
                    pass
        
        # Try filename pattern (anomaly_xxx.png vs normal_xxx.png)
        filename = img_path.stem.lower()
        if 'anomaly' in filename or 'abnormal' in filename:
            return 1
        return 0


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""
    
    def __init__(self, num_samples=1000, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
        np.random.seed(42)
        self.labels = np.random.randint(0, 2, num_samples)
        logger.info(f"Created synthetic dataset with {num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if label == 0:
            img = np.random.normal(0.5, 0.1, (3, self.img_size, self.img_size))
        else:
            img = np.random.normal(0.5, 0.2, (3, self.img_size, self.img_size))
            cx, cy = np.random.randint(50, 200, 2)
            img[:, max(0,cy-30):min(255,cy+30), max(0,cx-30):min(255,cx+30)] = 0.9
        
        img = np.clip(img, 0, 1).astype(np.float32)
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


def get_image_and_label_dirs(base_path):
    """
    Get image and label directories from a base path.
    
    Handles YOLO format where:
    - base_path = Dataset/train/
    - images are in Dataset/train/images/
    - labels are in Dataset/train/labels/
    
    Returns:
        (img_dir, label_dir) or (None, None) if not found
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        return None, None
    
    # Check for YOLO structure: base_path/images/ and base_path/labels/
    if (base_path / 'images').exists():
        img_dir = base_path / 'images'
        label_dir = base_path / 'labels'
        return img_dir, label_dir
    
    # Check if images are directly in base_path
    images = list(base_path.glob('*.png')) + list(base_path.glob('*.jpg'))
    if len(images) > 0:
        return base_path, None
    
    return None, None


def create_dataloaders(opt):
    """Create train and val dataloaders."""
    
    train_img_dir = None
    train_label_dir = None
    val_img_dir = None
    val_label_dir = None
    
    # Try to load dataset config
    if opt.data and Path(opt.data).exists():
        logger.info(f"Loading dataset config from: {opt.data}")
        with open(opt.data, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        # Get paths from config
        train_path = data_cfg.get('train', '')
        val_path = data_cfg.get('val', '')
        
        logger.info(f"Train path from config: {train_path}")
        logger.info(f"Val path from config: {val_path}")
        
        # Get actual image/label directories
        train_img_dir, train_label_dir = get_image_and_label_dirs(train_path)
        val_img_dir, val_label_dir = get_image_and_label_dirs(val_path)
        
        if train_img_dir:
            logger.info(f"Train images: {train_img_dir}")
            logger.info(f"Train labels: {train_label_dir}")
        if val_img_dir:
            logger.info(f"Val images: {val_img_dir}")
            logger.info(f"Val labels: {val_label_dir}")
    else:
        logger.warning(f"Data config not found: {opt.data}")
    
    # Create datasets
    if train_img_dir and train_img_dir.exists():
        logger.info(f"Creating training dataset from: {train_img_dir}")
        train_dataset = WSNDataset(train_img_dir, train_label_dir, opt.img_size, augment=True)
        
        if val_img_dir and val_img_dir.exists():
            logger.info(f"Creating validation dataset from: {val_img_dir}")
            val_dataset = WSNDataset(val_img_dir, val_label_dir, opt.img_size, augment=False)
        else:
            # Use portion of training for validation
            logger.warning("Validation set not found, using 20% of training data")
            total = len(train_dataset)
            val_size = int(0.2 * total)
            train_size = total - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
    else:
        logger.warning("=" * 60)
        logger.warning("USING SYNTHETIC DATASET - Real data not found!")
        logger.warning(f"Expected images at: {train_img_dir}")
        logger.warning("=" * 60)
        train_dataset = SyntheticDataset(num_samples=5000, img_size=opt.img_size)
        val_dataset = SyntheticDataset(num_samples=1000, img_size=opt.img_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward (force full model during training)
        output = model(images, force_full=True)
        
        # Compute multi-exit loss
        loss, loss_dict = criterion(output['all_logits'], labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = output['logits'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, device):
    """Validate with early exit enabled."""
    model.eval()
    model.reset_statistics()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward with early exit
            output = model(images, force_full=False)
            
            # Accuracy
            _, predicted = output['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    exit_stats = model.get_exit_statistics()
    
    return accuracy, exit_stats


def save_checkpoint(model, optimizer, epoch, accuracy, save_path):
    """Save checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, save_path)


# =============================================================================
# Main
# =============================================================================

def main(opt):
    print("=" * 70)
    print("EARLY EXIT WSN-YOLO TRAINING")
    print("Paper 3: Adaptive and Energy-Efficient Anomaly Detection")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create save directory
    save_dir = Path(opt.save_dir) / f"early_exit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Create dataloaders
    print("\n--- Loading Data ---")
    train_loader, val_loader = create_dataloaders(opt)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n--- Creating Model ---")
    model = EarlyExitModel(
        cfg=opt.cfg,
        ch=3,
        nc=2,
        exit1_layer=opt.exit1_layer,
        exit2_layer=opt.exit2_layer,
        exit1_threshold=opt.exit1_threshold,
        exit2_threshold=opt.exit2_threshold,
        num_classes=2
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = EarlyExitLoss(exit_weights=[0.3, 0.5, 1.0])
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 0.01)
    
    # Training loop
    print("\n--- Starting Training ---")
    best_accuracy = 0
    history = []
    
    for epoch in range(1, opt.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_acc, exit_stats = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch}/{opt.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Acc: {val_acc*100:.2f}%")
        print(f"  Exit Distribution: E1={exit_stats['exit_1_ratio']*100:.1f}%, "
              f"E2={exit_stats['exit_2_ratio']*100:.1f}%, "
              f"E3={exit_stats['exit_3_ratio']*100:.1f}%")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'exit_stats': exit_stats
        })
        
        # Save best
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, save_dir / 'best.pt')
            print(f"  ✓ Saved best model (acc={val_acc*100:.2f}%)")
        
        # Save last
        save_checkpoint(model, optimizer, epoch, val_acc, save_dir / 'last.pt')
    
    # Save history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    # Final report
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Accuracy: {best_accuracy*100:.2f}%")
    print(f"Model saved to: {save_dir}")
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(opt), f)
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--cfg', type=str, default='cfg/training/yolo_wsn_sso_optimized.yaml',
                        help='Model config file')
    parser.add_argument('--data', type=str, default='data/wsn_dataset.yaml',
                        help='Dataset config file')
    
    # Early exit
    parser.add_argument('--exit1-layer', type=int, default=9, help='Exit 1 layer index')
    parser.add_argument('--exit2-layer', type=int, default=18, help='Exit 2 layer index')
    parser.add_argument('--exit1-threshold', type=float, default=0.90, help='Exit 1 confidence threshold')
    parser.add_argument('--exit2-threshold', type=float, default=0.85, help='Exit 2 confidence threshold')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--img-size', type=int, default=256, help='Image size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='runs/train', help='Save directory')
    
    opt = parser.parse_args()
    
    main(opt)
