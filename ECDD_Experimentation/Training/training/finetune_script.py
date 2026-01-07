"""
ECDD Finetuning Script for LaDeDa-based ResNet50

Supports:
- Finetune 1 (Celeb-DF-v2 video frames)
- Finetune 2 (Face-filtered mixed images)

Follows ECDD protocol:
- Strict preprocessing (256x256, Lanczos, RGB, ImageNet norm)
- Attention pooling over patch logits
- Calibration-aware training
- Augmentations (JPEG compression, blur, resize chains)
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageOps
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ladeda_resnet import create_ladeda_model

# ============== Configuration ==============

CONFIGS = {
    "finetune1": {
        "name": "finetune1_celeb_df",
        "dataset_path": Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation\ECDD_Training_Data\processed\splits\finetune1"),
        "epochs": 15,
        "batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "freeze_layers": ["conv1", "layer1"],  # Moderate freeze
        "augment": True,
        "description": "Celeb-DF-v2 video frames (balanced)"
    },
    "finetune2": {
        "name": "finetune2_face_filtered",
        "dataset_path": Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation\ECDD_Training_Data\processed\splits\finetune2"),
        "epochs": 20,
        "batch_size": 8,  # Smaller dataset, smaller batch
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "freeze_layers": ["conv1", "layer1"],
        "augment": True,
        "description": "Face-filtered mixed images"
    }
}

# Preprocessing constants (ECDD-locked)
TARGET_SIZE = (256, 256)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============== Dataset ==============

class FinetuneDataset(Dataset):
    """Dataset for finetuning with ECDD-compliant preprocessing."""
    
    def __init__(self, data_dir: Path, split: str = "train", augment: bool = True):
        self.data_dir = Path(data_dir) / split
        self.augment = augment and (split == "train")
        
        self.images = []
        self.labels = []
        
        # Load real images
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for f in real_dir.glob("*.jpg"):
                self.images.append(f)
                self.labels.append(0)
        
        # Load fake images
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for f in fake_dir.glob("*.jpg"):
                self.images.append(f)
                self.labels.append(1)
        
        print(f"Loaded {len(self.images)} images for {split} (Real: {self.labels.count(0)}, Fake: {self.labels.count(1)})")
    
    def __len__(self):
        return len(self.images)
    
    def _augment(self, img: Image.Image) -> Image.Image:
        """Apply training augmentations per ECDD protocol."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random JPEG compression (simulate social media)
        if random.random() > 0.5:
            import io
            quality = random.randint(50, 95)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
            img.load()  # Force load before buffer closes
        
        # Random brightness/contrast (slight)
        if random.random() > 0.7:
            from PIL import ImageEnhance
            factor = random.uniform(0.9, 1.1)
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        return img
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess (ECDD-locked)
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # Fix EXIF orientation
        img = img.convert('RGB')
        
        # Augmentation (training only)
        if self.augment:
            img = self._augment(img)
        
        # Resize with Lanczos (ECDD-locked)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # To tensor and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # HWC -> CHW
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return img_tensor, label_tensor


# ============== Training ==============

def compute_metrics(outputs, labels, threshold=0.5):
    """Compute accuracy, precision, recall, F1."""
    probs = torch.sigmoid(outputs).cpu().numpy()
    preds = (probs > threshold).astype(int)
    labels = labels.cpu().numpy().astype(int)
    
    correct = (preds == labels).sum()
    accuracy = correct / len(labels)
    
    # For fake class (label=1)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            pooled_logit, patch_logits, attention = model(images)
            loss = criterion(pooled_logit.squeeze(), labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        all_outputs.append(pooled_logit.squeeze().detach())
        all_labels.append(labels)
        
        pbar.set_postfix({'loss': loss.item()})
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            pooled_logit, _, _ = model(images)
            loss = criterion(pooled_logit.squeeze(), labels)
            
            total_loss += loss.item()
            all_outputs.append(pooled_logit.squeeze())
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def run_finetuning(config_name: str, output_dir: Path = None):
    """Run finetuning for specified config."""
    
    config = CONFIGS[config_name]
    print("=" * 60)
    print(f"ECDD Finetuning: {config['name']}")
    print(f"Description: {config['description']}")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "checkpoints" / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Datasets
    train_dataset = FinetuneDataset(config['dataset_path'], split="train", augment=config['augment'])
    val_dataset = FinetuneDataset(config['dataset_path'], split="val", augment=False)
    test_dataset = FinetuneDataset(config['dataset_path'], split="test", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # Model
    model = create_ladeda_model(pretrained=True, freeze_layers=config['freeze_layers'])
    model = model.to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler()
    
    # Training loop
    best_val_f1 = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"  -> Saved best model (Val F1: {best_val_f1:.4f})")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    print(f"       Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    # Save final results
    results = {
        'config': config['name'],
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")
    return results


# ============== Main ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECDD LaDeDa Finetuning")
    parser.add_argument('--config', type=str, choices=['finetune1', 'finetune2'], required=True,
                        help="Which config to use")
    parser.add_argument('--output', type=str, default=None,
                        help="Output directory for checkpoints")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    run_finetuning(args.config, output_dir)
