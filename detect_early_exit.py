"""
Detection Script for Early Exit WSN-YOLO
========================================
Paper 3: Adaptive and Energy-Efficient Anomaly Detection in WSNs

This script runs inference with adaptive computation:
- Easy images exit early (save energy)
- Hard images use full model (maintain accuracy)

Place this file in your project ROOT directory.

Usage:
    python detect_early_exit.py --weights runs/train/early_exit/best.pt --source path/to/images/

Author: Maidul Hasan Masum
University of Malaya
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn.functional as F

# Setup path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo import EarlyExitModel
from models.early_exit import EnergyCostModel, ExitStatistics


class EarlyExitDetector:
    """
    Detector with early exit capability.
    
    Automatically uses less computation for easy images.
    """
    
    def __init__(self, weights, cfg, device='cuda', 
                 exit1_threshold=0.90, exit2_threshold=0.85):
        """
        Initialize detector.
        
        Args:
            weights: Path to trained weights
            cfg: Path to model config
            device: Device to run on
            exit1_threshold: Confidence threshold for exit 1
            exit2_threshold: Confidence threshold for exit 2
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {weights}...")
        self.model = EarlyExitModel(
            cfg=cfg,
            ch=3,
            nc=2,
            exit1_threshold=exit1_threshold,
            exit2_threshold=exit2_threshold,
            num_classes=2
        )
        
        # Load weights
        if Path(weights).exists():
            ckpt = torch.load(weights, map_location=self.device)
            if 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(ckpt)
            print(f"Loaded weights from {weights}")
        else:
            print(f"Warning: Weights not found at {weights}, using random initialization")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Cost model
        self.cost_model = EnergyCostModel()
        
        # Statistics
        self.stats = ExitStatistics()
        
        # Class names
        self.class_names = ['Normal', 'Anomaly']
        
        print(f"Model loaded on {self.device}")
        print(f"Exit thresholds: E1={exit1_threshold}, E2={exit2_threshold}")
    
    def preprocess(self, image, img_size=256):
        """Preprocess image for inference."""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            raise ValueError("Failed to load image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (img_size, img_size))
        
        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(image).unsqueeze(0)  # Add batch dim
        
        return image
    
    def detect(self, image, img_size=256):
        """
        Run detection on single image.
        
        Args:
            image: Image path or numpy array
            img_size: Input size
        
        Returns:
            dict with prediction, confidence, exit_id, energy, latency
        """
        # Preprocess
        img_tensor = self.preprocess(image, img_size)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(img_tensor, force_full=False)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get prediction
        logits = output['logits']
        probs = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)
        
        exit_id = output['exit_id']
        
        # Get costs
        energy = self.cost_model.get_energy(exit_id)
        latency = self.cost_model.get_latency(exit_id)
        
        # Update statistics
        self.stats.update(exit_id)
        
        result = {
            'prediction': predicted.item(),
            'class_name': self.class_names[predicted.item()],
            'confidence': confidence.item(),
            'exit_id': exit_id,
            'exit_name': ['Exit 1 (Early)', 'Exit 2 (Middle)', 'Full Model'][exit_id],
            'energy_mj': energy,
            'latency_ms': latency,
            'actual_time_ms': inference_time,
            'probs': probs.cpu().numpy()[0]
        }
        
        return result
    
    def detect_batch(self, images, img_size=256):
        """Run detection on multiple images."""
        results = []
        for img in images:
            result = self.detect(img, img_size)
            results.append(result)
        return results
    
    def get_statistics(self):
        """Get exit statistics."""
        return self.stats.get_report()
    
    def print_statistics(self):
        """Print exit statistics."""
        self.stats.print_report()
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats.reset()
    
    def set_thresholds(self, t1, t2):
        """Set exit thresholds."""
        self.model.set_thresholds(t1, t2)


def run_detection(opt):
    """Run detection on images."""
    print("=" * 70)
    print("EARLY EXIT WSN-YOLO DETECTION")
    print("Adaptive Computation - Less energy for easy images")
    print("=" * 70)
    
    # Create detector
    detector = EarlyExitDetector(
        weights=opt.weights,
        cfg=opt.cfg,
        device='cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu',
        exit1_threshold=opt.exit1_threshold,
        exit2_threshold=opt.exit2_threshold
    )
    
    # Get image paths
    source = Path(opt.source)
    if source.is_file():
        img_paths = [source]
    elif source.is_dir():
        img_paths = sorted(list(source.glob('*.png')) + 
                          list(source.glob('*.jpg')) + 
                          list(source.glob('*.jpeg')))
    else:
        print(f"Error: Source not found: {source}")
        return
    
    print(f"\nProcessing {len(img_paths)} images...")
    print("-" * 70)
    
    # Process each image
    for i, img_path in enumerate(img_paths):
        result = detector.detect(str(img_path), opt.img_size)
        
        print(f"\nImage {i+1}: {img_path.name}")
        print(f"  Prediction: {result['class_name']} ({result['confidence']*100:.1f}%)")
        print(f"  Exit Used: {result['exit_name']}")
        print(f"  Energy: {result['energy_mj']:.2f} mJ")
        print(f"  Latency: {result['actual_time_ms']:.1f} ms")
        
        # Save result if output dir specified
        if opt.output:
            output_dir = Path(opt.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and annotate image
            img = cv2.imread(str(img_path))
            
            # Add text
            label = f"{result['class_name']} ({result['confidence']*100:.1f}%)"
            exit_label = f"Exit: {result['exit_id']+1}, Energy: {result['energy_mj']:.1f}mJ"
            
            color = (0, 255, 0) if result['prediction'] == 0 else (0, 0, 255)
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, exit_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save
            cv2.imwrite(str(output_dir / img_path.name), img)
    
    # Print statistics
    print("\n")
    detector.print_statistics()
    
    # Calculate savings
    report = detector.get_statistics()
    if report:
        baseline_energy = 1.9 * report['total_samples']
        actual_energy = report['avg_energy_mj'] * report['total_samples']
        saved_energy = baseline_energy - actual_energy
        
        print(f"\nENERGY SUMMARY:")
        print(f"  Without early exit: {baseline_energy:.2f} mJ total")
        print(f"  With early exit: {actual_energy:.2f} mJ total")
        print(f"  Energy saved: {saved_energy:.2f} mJ ({report['energy_savings']*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--weights', type=str, required=True, help='Path to weights')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolo_wsn_sso_optimized.yaml',
                        help='Model config')
    
    # Source
    parser.add_argument('--source', type=str, required=True, help='Image path or directory')
    parser.add_argument('--img-size', type=int, default=256, help='Input size')
    
    # Thresholds
    parser.add_argument('--exit1-threshold', type=float, default=0.90, help='Exit 1 threshold')
    parser.add_argument('--exit2-threshold', type=float, default=0.85, help='Exit 2 threshold')
    
    # Output
    parser.add_argument('--output', type=str, default='', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    
    opt = parser.parse_args()
    
    run_detection(opt)
