# Adaptive LightYolo-SSO: Energy-Efficient Anomaly Detection in Wireless Sensor Networks via Dynamic Early Exit Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An adaptive inference framework that extends the SSO-optimized [LightYolo-SSO](https://github.com/mh-masum6875/LightYolo-SSO) with dynamic early exit capability for energy-efficient anomaly detection in resource-constrained Wireless Sensor Networks. Based on [YOLOv7](https://github.com/WongKinYiu/yolov7).

The model is a single trained network that runs in two modes. With early exit enabled it terminates inference early for confident samples; with early exit disabled it runs the full network as a baseline reference. Both modes share the same weights.

## Key Results

Measured on the 3,109-sample IBRL test set, one sample at a time.

| Metric | Adaptive (early exit on) | Full depth (early exit off) |
|--------|--------------------------|------------------------------|
| Accuracy | 99.55% | 99.52% |
| Precision | 99.66% | 99.61% |
| Recall | 99.66% | 99.66% |
| F1-score | 99.66% | 99.64% |
| Average compute | 92.2 M MACs | 261.2 M MACs |
| Cost reduction | **64.7%** | — |

- Total parameters: **1,395,540** (1,370,478 backbone + 25,062 exit branches, a **1.8% overhead**)
- Per-sample exit distribution: **98.6%** at Exit 1, **0.7%** at Exit 2, **0.7%** at the full network
- Energy and latency reduce in proportion to the compute reduction under standard linear-scaling assumptions. These are FLOP-proportional estimates; hardware-measured energy is planned future work.

## Architecture

```
Input Heatmap (256×256)
        │
        ▼
┌──────────────────────────────┐
│  Stage 1 (Layers 0-14)       │──► Exit 1 (θ₁ = 0.90) ──► 98.6% of samples
│  64-channel feature, 32×32   │    branch: 2,146 params
│                              │    path cost: 181.5 M FLOPs (34.8%)
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 2 (Layers 15-21)      │──► Exit 2 (θ₂ = 0.85) ──► 0.7% of samples
│  120-channel feature, 16×16  │    branch: 7,874 params
│                              │    path cost: 239.9 M FLOPs (45.9%)
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 3 (Layers 22-end)     │──► Final Classifier ──► 0.7% of samples
│  232-channel feature         │    head: 15,042 params
│                              │    path cost: 522.4 M FLOPs (100%)
└──────────────────────────────┘
```

The backbone is the LightYolo-SSO width-0.6 model. Exit branches use global average pooling, a fully connected layer with ReLU and dropout, and a final linear layer producing the two class logits.

## Project Structure

```
Adaptive-LightYolo-SSO/
├── cfg/                                # Model configurations
│   └── training/
│       └── yolo_wsn_sso_optimized.yaml        # LightYolo-SSO backbone (width 0.6)
├── data/                               # Dataset configs
│   └── wsn_dataset.yaml
├── models/                             # Model implementations
│   ├── common.py
│   ├── yolo.py                         # EarlyExitModel class
│   └── early_exit.py                   # Early exit components
├── utils/                              # Utility functions
├── SSO/                                # SSO optimization code
├── runs/                               # Training outputs
│   └── EarlyExit_PaperB_23rdMay2026/
│       └── early_exit_20260523_190210/ # best.pt, history.json
├── scripts/                            # Additional scripts
├── tools/                              # Analysis tools
├── train_early_exit.py                 # Training script
├── test_early_exit.py                  # Evaluation script
├── detect_early_exit.py                # Inference script
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/mh-masum6875/Adaptive-LightYolo-SSO.git
cd Adaptive-LightYolo-SSO
pip install -r requirements.txt
```

## Dataset

IBRL (Intel Berkeley Research Laboratory) dataset:

| Characteristic | Value |
|----------------|-------|
| Total Images | 31,067 (256×256 grayscale) |
| Classes | 2 (Normal, Anomaly) |
| Training Set | 24,853 (80%) |
| Validation Set | 3,105 (10%) |
| Test Set | 3,109 (10%) |
| Class distribution | 33.6% Normal, 66.4% Anomaly |

### Dataset Structure

```
Dataset/
├── train/
│   └── images/
│       ├── normal_0001.png
│       ├── anomaly_0001.png
│       └── ...
├── valid/
│   └── images/
└── test/
    └── images/
```

## Training

The backbone is initialized from the trained LightYolo-SSO weights, then the exit branches are attached and the whole model is trained end to end.

```bash
python train_early_exit.py \
    --cfg cfg/training/yolo_wsn_sso_optimized.yaml \
    --data data/wsn_dataset.yaml \
    --init-weights runs/train/Updated_wsn_sso_transfer/weights/best.pt \
    --epochs 150 \
    --batch-size 16 \
    --img-size 256 \
    --exit1-threshold 0.90 \
    --exit2-threshold 0.85 \
    --save-dir runs/EarlyExit_PaperB
```

Exit branches are placed at layers 14 and 21. Training uses AdamW with cosine annealing, an initial learning rate of 0.001, weight decay of 0.0001, gradient clipping at a max norm of 10, and multi-exit loss weights of [0.3, 0.5, 1.0].

## Inference

```bash
python detect_early_exit.py \
    --weights runs/EarlyExit_PaperB_23rdMay2026/early_exit_20260523_190210/best.pt \
    --cfg cfg/training/yolo_wsn_sso_optimized.yaml \
    --source path/to/images \
    --img-size 256
```

## Evaluation

```bash
python test_early_exit.py \
    --weights runs/EarlyExit_PaperB_23rdMay2026/early_exit_20260523_190210/best.pt \
    --cfg cfg/training/yolo_wsn_sso_optimized.yaml \
    --data data/wsn_dataset.yaml
```

## Threshold Configurations

Measured by re-running inference at different thresholds on the trained model. Accuracy stays essentially flat while the exit distribution and cost shift.

| Configuration | θ₁ | θ₂ | Accuracy | Cost Reduction |
|---------------|-----|-----|----------|----------------|
| Aggressive | 0.80 | 0.75 | 99.55% | 65.0% |
| **Default** | **0.90** | **0.85** | **99.55%** | **64.7%** |
| Conservative | 0.95 | 0.90 | 99.55% | 64.5% |

Tightening the thresholds beyond the conservative setting routes more samples to the deeper exits while accuracy stays at 99.52% to 99.55%, confirming that all three exit points are functional.

## Training Behavior

Training converged smoothly over 150 epochs with no overfitting. The model reached high accuracy within the first ten epochs, reflecting the backbone transfer from LightYolo-SSO, then settled to a stable plateau.

| Checkpoint | Train Acc | Val Acc |
|------------|-----------|---------|
| Epoch 1 | 98.25% | ~99.2% |
| Best (Epoch 72) | 99.77% | 99.74% |
| Final (Epoch 150) | 99.94% | 99.68% |

The reported test results use the best checkpoint at `early_exit_20260523_190210/best.pt`.

## Cost Analysis

Per-exit compute is measured directly from the trained model with a FLOP-profiling tool. Average per-sample cost follows from the per-sample exit distribution.

```
Average MACs = (0.986 × 90.8) + (0.007 × 120.0) + (0.007 × 261.2) = 92.2 M MACs
Cost Reduction = (261.2 − 92.2) / 261.2 = 64.7%
```

This is a FLOP-proportional estimate of efficiency. Absolute energy and latency depend on the deployment platform and are planned as hardware-validation work.

## Related Work

This project builds upon:

1. **Our Review Paper:** [Toward Efficient and Accurate Anomaly Detection in Wireless Sensor Networks: A Review](https://ieeexplore.ieee.org/document/11218714) - IEEE Sensors Journal, 2025
2. **Our LightYolo-SSO:** [LightYolo-SSO](https://github.com/mh-masum6875/LightYolo-SSO) - SSO-optimized lightweight detection backbone
3. **YOLOv7:** [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

## Citation

If you use this code, please cite our review paper:

```bibtex
@article{masum2025review,
  title={Toward Efficient and Accurate Anomaly Detection in Wireless Sensor
         Networks: A Review},
  author={Masum, Maidul Hasan and Raj, Bryan and Showmik, Shakib Sahariar},
  journal={IEEE Sensors Journal},
  year={2025},
  doi={10.1109/JSEN.2025.3566188}
}
```

## References

1. Shehadeh, H.A., Ahmadi, H., Hoare, A., Idris, M.Y.I. "Sperm swarm optimization algorithm for optimizing wireless sensor network challenges." *IEEE Access*, vol. 9, pp. 65169-65181, 2021.

2. Wang, C.Y., Bochkovskiy, A., Liao, H.Y.M. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7464-7475, 2023.

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

**Maidul Hasan Masum**
University of Malaya
Email: mhmasum6875@um.edu.my
