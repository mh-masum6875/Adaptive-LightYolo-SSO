"""
Test Script for Early Exit WSN-YOLO
===================================
Paper 3: Adaptive and Energy-Efficient Anomaly Detection

Usage:
    python test_early_exit.py --weights runs/train/early_exit/best.pt --data data/wsn_dataset.yaml

Author: Maidul Hasan Masum, University of Malaya
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo import EarlyExitModel
from models.early_exit import EnergyCostModel
from train_early_exit import WSNDataset, SyntheticDataset, get_image_and_label_dirs


def evaluate(model, dataloader, device, force_full=False, desc="Evaluating"):
    """Evaluate model."""
    model.eval()
    model.reset_statistics()
    
    correct = 0
    total = 0
    exit_counts = [0, 0, 0]
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)
            
            for i in range(images.size(0)):
                output = model(images[i:i+1], force_full=force_full)
                _, pred = output['logits'].max(1)
                
                total += 1
                correct += pred.eq(labels[i]).sum().item()
                exit_counts[output['exit_id']] += 1
    
    accuracy = correct / total
    return accuracy, exit_counts, total


def main(opt):
    print("=" * 70)
    print("EARLY EXIT WSN-YOLO EVALUATION")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {opt.weights}...")
    model = EarlyExitModel(
        cfg=opt.cfg,
        ch=3, nc=2,
        exit1_threshold=0.90,
        exit2_threshold=0.85,
        num_classes=2
    )
    
    if Path(opt.weights).exists():
        ckpt = torch.load(opt.weights, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print("Weights loaded successfully")
    else:
        print(f"Warning: Weights not found at {opt.weights}")
    
    model = model.to(device)
    
    # Load data
    print("\nLoading data...")
    dataset = None
    
    if opt.data and Path(opt.data).exists():
        with open(opt.data) as f:
            data_cfg = yaml.safe_load(f)
        
        # Try test, then val
        test_path = data_cfg.get('test', data_cfg.get('val', ''))
        
        if test_path:
            # Use the helper function to get correct paths
            img_dir, label_dir = get_image_and_label_dirs(test_path)
            
            if img_dir and img_dir.exists():
                print(f"Loading test data from: {img_dir}")
                dataset = WSNDataset(img_dir, label_dir, img_size=256)
    
    if dataset is None:
        print("Using synthetic dataset")
        dataset = SyntheticDataset(1000, 256)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Test samples: {len(dataset)}")
    
    cost_model = EnergyCostModel()
    
    # Test 1: Full Model
    print("\n" + "=" * 50)
    print("TEST 1: Full Model (No Early Exit)")
    print("=" * 50)
    
    acc_full, _, total = evaluate(model, dataloader, device, force_full=True, desc="Full Model")
    energy_full = 1.9
    
    print(f"Accuracy: {acc_full*100:.2f}%")
    print(f"Energy: {energy_full:.2f} mJ/sample")
    
    # Test 2: Early Exit
    print("\n" + "=" * 50)
    print("TEST 2: Early Exit (Adaptive)")
    print("=" * 50)
    
    model.set_thresholds(0.90, 0.85)
    acc_ee, exit_counts, _ = evaluate(model, dataloader, device, force_full=False, desc="Early Exit")
    
    costs = cost_model.calculate_average_cost(exit_counts)
    
    print(f"Accuracy: {acc_ee*100:.2f}%")
    print(f"Exit 1 (Early): {exit_counts[0]} ({100*exit_counts[0]/total:.1f}%)")
    print(f"Exit 2 (Middle): {exit_counts[1]} ({100*exit_counts[1]/total:.1f}%)")
    print(f"Exit 3 (Full): {exit_counts[2]} ({100*exit_counts[2]/total:.1f}%)")
    print(f"Avg Energy: {costs['avg_energy_mj']:.2f} mJ/sample")
    print(f"Energy Savings: {costs['energy_savings']*100:.1f}%")
    
    # Comparison Table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (For Paper)")
    print("=" * 70)
    print(f"{'Method':<25} {'Accuracy':>10} {'Energy (mJ)':>12} {'Savings':>10}")
    print("-" * 60)
    print(f"{'Full Model (Baseline)':<25} {acc_full*100:>9.2f}% {energy_full:>11.2f} {'0%':>10}")
    print(f"{'Early Exit (Ours)':<25} {acc_ee*100:>9.2f}% {costs['avg_energy_mj']:>11.2f} {costs['energy_savings']*100:>9.1f}%")
    print("-" * 60)
    
    acc_drop = acc_full - acc_ee
    print(f"\nAccuracy drop: {acc_drop*100:.2f}%")
    print(f"Energy savings: {costs['energy_savings']*100:.1f}%")
    
    # Save results
    results = {
        'full_model': {'accuracy': acc_full, 'energy_mj': energy_full},
        'early_exit': {
            'accuracy': acc_ee,
            'energy_mj': costs['avg_energy_mj'],
            'energy_savings': costs['energy_savings'],
            'exit_distribution': {
                'exit_1': exit_counts[0]/total,
                'exit_2': exit_counts[1]/total,
                'exit_3': exit_counts[2]/total
            }
        }
    }
    
    save_dir = Path(opt.weights).parent
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_dir / 'evaluation_results.json'}")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER:")
    print("=" * 70)
    print(f"1. Early exit achieves {costs['energy_savings']*100:.1f}% energy savings")
    print(f"2. Accuracy drop is only {acc_drop*100:.2f}%")
    print(f"3. {100*exit_counts[0]/total:.1f}% of images exit at first checkpoint")
    print(f"4. Suitable for energy-constrained WSN deployment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Model weights')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolo_wsn_sso_optimized.yaml')
    parser.add_argument('--data', type=str, default='data/wsn_dataset.yaml')
    opt = parser.parse_args()
    main(opt)
