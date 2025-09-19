# ScanNet-PyTorch

This repository is a **PyTorch migration of the original ScanNet model** for structure-based protein binding site prediction.  
The project is part of the **Master’s Thesis of Anagha Siddhi** at **Oklahoma State University, Stillwater** (Department of Computer Science, ILEAD Lab).  

This repo provides:
- Preprocessing pipeline for the **BioLiP** dataset (`preprocess_biolip.py`)  
- Training pipeline with **ScanNet** (`train_pytorch.py`)  
- Support for distributed training (DDP), early stopping, focal loss, hard negative mining, and exponential moving average (EMA).

##  Repository Structure

ScanNet-PyTorch/
│── scannet_pytorch/
│   ├── network/           # ScanNet model (attention, embeddings, utils)
│   ├── preprocessing/     # Preprocessing scripts for BioLiP
│   ├── scripts/           # Utility + evaluation scripts
│   ├── utilities/         # Helpers: data loaders, training utils, I/O
│   ├── train_pytorch.py   # Main training entrypoint
│   └── __init__.py
│
│── configs/               # Example configs
│── results/               # Metrics, logs, checkpoints (ignored in Git)
│── data/                  # Processed splits + ids (ignored in Git)
│── archive/               # Legacy scripts (not required)
│── README.md
│── .gitignore
│── LICENSE

## Installation
1. Clone the repo:
   ```bash
   git clone git@github.com:anaghasiddhi/Scannet_pytorch.git
   cd Scannet_pytorch
```
2. Create environment:

```
conda create -n scannet_pytorch python=3.10
conda activate scannet_pytorch
pip install -r requirements.txt
```
Or use the provided scannet_pytorch_env.yml with conda:

```
conda env create -f scannet_pytorch/scannet_pytorch_env.yml
conda activate scannet_pytorch
```

## Data Preprocessing

### Inputs
- **BioLiP dataset** (CIF structures + annotations).  
- Provide paths via environment variables or command line:
```
  - `BIOLIP_DIR` → root BioLiP folder  
  - `BIOLIP_STRUCTURES` → CIF structures directory  
  - `BIOLIP_ANN` → BioLiP annotation CSV  
```
### Run Preprocessing
```bash
python scannet_pytorch/preprocessing/preprocess_biolip.py \
    --cif-root /path/to/biolip/structures \
    --output-dir data \
    --num-workers 8
```

### Outputs:

data/train.data/*.npy → feature quartets

data/train_labels.pkl → labels

data/train_processed_ids.txt → processed chain IDs
(similar for val and test).

## Training

### Run single-GPU training:
```bash
python scannet_pytorch/train_pytorch.py \
    --epochs 30 \
    --batch_size 8 \
    --log_dir results/logs_run1 \
    --save_dir checkpoints \
    --use_processed_ids \
    --train_ids data/train_processed_ids.txt \
    --val_ids data/val_processed_ids.txt \
    --test_ids data/test_processed_ids.txt
```
### Run distributed (multi-GPU) training:

```bash

torchrun --nproc_per_node=4 scannet_pytorch/train_pytorch.py \
    --epochs 30 \
    --batch_size 8 \
    --dist \
    --use_processed_ids
```

---

## Evaluation


Run evaluation with a saved checkpoint:
```bash
python scannet_pytorch/train_pytorch.py \
    --eval_only \
    --resume checkpoints/best_model_val.pt \
    --use_processed_ids \
    --val_ids data/val_processed_ids.txt \
    --test_ids data/test_processed_ids.txt
```
### Outputs:

results/logs_run1/pr_curve_val.png

results/logs_run1/ops_val.json



---

## Features

- **BioLiP preprocessing** with multiprocessing, alignment, and quality checks.  
- **ScanNet model** with attention-based architecture.  
- **Training loop** with:
  - Early stopping
  - Learning rate schedulers (cosine, plateau, one-cycle)
  - BCE or focal loss
  - Hard negative mining
  - Exponential Moving Average (EMA) of weights
- **Evaluation** with AUROC, PR-AUC, P@R, Top-K metrics, and PR curve plotting.

## References
This program is a migration of the following:  

Tubiana, J., Schneidman-Duhovny, D., & Wolfson, H. J. (2022). **ScanNet: An interpretable geometric deep learning model for structure-based protein binding site prediction.** *Nature Methods*.  

If you use this program, please cite the above work.  

---
For questions, please reach out to **anagha.siddhi@okstate.edu**  

This work is part of the **Master’s Thesis** of *Anagha Siddhi* at **Oklahoma State University, Stillwater**,  
Department of Computer Science, under the **ILEAD Lab**.  

