# Adaptive LightYolo-SSO: Energy-Efficient Anomaly Detection in Wireless Sensor Networks via Dynamic Early Exit Inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An adaptive inference framework that extends the SSO-optimized [LightYolo-SSO](https://github.com/mh-masum6875/-LightYolo-SSO) with dynamic early exit capability for energy-efficient anomaly detection in resource-constrained Wireless Sensor Networks. Based on [YOLOv7](https://github.com/WongKinYiu/yolov7).

## Key Achievements

| Metric | Adaptive LightYolo-SSO | LightYolo-SSO (Baseline) | Improvement |
|--------|------------------------|--------------------------|-------------|
| Parameters | 81,670 | 67,840 | +20.4% overhead |
| Validation Accuracy | 99.58% | 99.68% | -0.10% (negligible) |
| Average Energy | 0.619 mJ | 1.90 mJ | **67.4% reduction** |
| Average Latency | 24.16 ms | 80 ms | **69.8% reduction** |
| Exit 1 Usage | 72.7% | - | Early termination |
| Exit 2 Usage | 19.1% | - | Mid termination |
| Exit 3 Usage | 8.2% | - | Full computation |

## Architecture

```
Input Image (256×256)
        │
        ▼
┌───────────────────────┐
│  Stage 1 (Layers 0-9) │──────► Exit 1 (θ₁=0.90) ──► 72.7% samples
│     5,248 params      │        Energy: 0.4 mJ | Latency: 15 ms
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Stage 2 (Layers 10-18)│──────► Exit 2 (θ₂=0.85) ──► 19.1% samples
│    18,944 params      │        Energy: 0.9 mJ | Latency: 35 ms
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│ Stage 3 (Layers 19-71)│──────► Exit 3 (Full) ──► 8.2% samples
│    43,648 params      │        Energy: 1.9 mJ | Latency: 80 ms
└───────────────────────┘
```

## Project Structure

```
Adaptive-LightYolo-SSO/
├── cfg/                                # Model configurations
│   └── training/
│       └── yolo_wsn_sso_optimized.yaml # SSO-optimized backbone
├── data/                               # Dataset configs
│   └── wsn_dataset.yaml               
├── models/                             # Model implementations
│   ├── common.py                      
│   ├── yolo.py                        # EarlyExitModel class
│   └── early_exit.py                  # Early exit components
├── utils/                              # Utility functions
├── SSO/                                # SSO optimization code
├── runs/                               # Training outputs
│   └── train/EarlyExit_WSN_YOLO/      
├── scripts/                            # Additional scripts
├── tools/                              # Analysis tools
├── train_early_exit.py                 # Training script
├── test_early_exit.py                  # Validation script
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

```bash
python train_early_exit.py \
    --cfg cfg/training/yolo_wsn_sso_optimized.yaml \
    --data data/wsn_dataset.yaml \
    --epochs 150 \
    --batch-size 16 \
    --img-size 256 \
    --exit1-threshold 0.90 \
    --exit2-threshold 0.85 \
    --name EarlyExit_WSN_YOLO
```

## Inference

```bash
python detect_early_exit.py \
    --weights runs/train/EarlyExit_WSN_YOLO/weights/best.pt \
    --source path/to/images \
    --img-size 256
```

## Validation

```bash
python test_early_exit.py \
    --weights runs/train/EarlyExit_WSN_YOLO/weights/best.pt \
    --data data/wsn_dataset.yaml
```

## Threshold Configurations

| Configuration | θ₁ | θ₂ | Accuracy | Energy Savings |
|---------------|-----|-----|----------|----------------|
| Conservative | 0.95 | 0.90 | 99.64% | 58.9% |
| **Default** | **0.90** | **0.85** | **99.58%** | **67.4%** |
| Aggressive | 0.85 | 0.80 | 99.45% | 72.6% |

## Training Results

| Epoch | Train Loss | Train Acc | Val Acc | Exit 1 % | Exit 2 % | Exit 3 % |
|-------|------------|-----------|---------|----------|----------|----------|
| 1 | 0.2080 | 97.12% | 98.90% | 39.7% | 39.2% | 21.1% |
| 50 | 0.0278 | 99.57% | 99.55% | 70.3% | 20.1% | 9.6% |
| 100 | 0.0155 | 99.84% | 99.61% | 71.7% | 19.6% | 8.7% |
| 150 | 0.0120 | 99.94% | 99.58% | 72.7% | 19.1% | 8.2% |

## Energy Analysis

```
Average Energy = (0.727 × 0.4) + (0.191 × 0.9) + (0.082 × 1.9) = 0.619 mJ
Energy Savings = (1.90 - 0.619) / 1.90 = 67.4%
```

## Related Work

This project builds upon:

1. **Our Review Paper:** [Toward Efficient and Accurate Anomaly Detection in Wireless Sensor Networks: A Review](https://ieeexplore.ieee.org/document/11218714) - IEEE Sensors Journal, 2025
2. **Our LightYolo-SSO:** [LightYolo-SSO](https://github.com/mh-masum6875/-LightYolo-SSO) - 98.91% parameter reduction
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
