"""
Dual-backbone fusion model for disaster-severity classification.
Core implementation is redacted. Structure & design decisions are visible.

Pipeline:
    [Weak-aug image]   ──► ViT-B/16 Stem A ──┐
                                               ├──► ConcatDeltaProd Fusion ──► DomainFiLM ──► Head
    [Strong-aug image] ──► ViT-B/16 Stem B ──┘
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

import os, json, time, random, hashlib, warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import timm

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    balanced_accuracy_score, roc_auc_score, average_precision_score, log_loss,
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# Device
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CFG = {
    # ── data ──
    "bright_dir": "<PATH_TO_BRIGHT_DATASET>",
    "crisis_dir": None,
    "classes": ["minor", "moderate", "severe"],
    "class_to_idx": {"minor": 0, "moderate": 1, "severe": 2},
    "img_size": 512,
    "use_subset": False,
    "train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15,
    "seed": 42,
    # ── loader ──
    "batch_size": 8,
    "num_workers": 0,
    # ── model ──
    "stems": ("vit_b16", "vit_b16"),
    "fusion": "concat_delta_prod",       # key design choice
    "proj_dim": 512,
    "domain_film_dim": 64,
    "num_classes": 3,
    "num_domains": 2,
    "dropout": 0.2,
    # ── optimisation ──
    "epochs_full": 50,
    "lr": 1e-4,
    "lr_vit_mult": 0.05,
    "lr_resnet_mult": 0.1,
    "weight_decay": 1e-4,
    "scheduler": "plateau",
    "pct_start": 0.2,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    # ── losses ──
    "loss_type": "focal",
    "label_smoothing": 0.1,
    "focal_gamma": 2.0,
    "use_consistency": True,
    "consistency_lambda": 0.5,
    # ── misc ──
    "amp_eval_off": True,
    "eps": 1e-6,
    "out_dir_root": "./runs_dualstem_fusion",
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(CFG["seed"])


# ══════════════════════════════════════════════════════════════════════════════
# 2. UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file for cross-domain deduplication."""
    # ... [implementation hidden]
    pass


def read_image_as_rgb(path: str) -> np.ndarray:
    """
    Read any image format (TIFF / JPG / PNG) as uint8 RGB.
    TIFF multi-channel images are percentile-stretched (p1–p99) before conversion.
    """
    # ... [implementation hidden]
    pass


def jpeg_reencode(img, quality: int = 70):
    """Re-encode image as JPEG to simulate compression artefacts (strong augmentation)."""
    # ... [implementation hidden]
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET SCANNING, DEDUP & STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════════════════════
# Outputs (from full run):
#   Found: BRIGHT=6705 → after SHA-256 dedup → 6682 unique samples
#   Train: 4675  |  Val: 1001  |  Test: 1006

def scan_domain(root, domain_id: int, classes: list) -> list:
    """Walk dataset directory, collect (path, label, domain_id) tuples."""
    # ... [implementation hidden]
    pass


def stratified_split(items, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Stratified split by (label × domain) key.
    Guarantees every (class, domain) combination appears in each split.
    Returns: train_items, val_items, test_items
    """
    # ... [implementation hidden]
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASET — DUAL-VIEW AUGMENTATION & BALANCED SAMPLER
# ══════════════════════════════════════════════════════════════════════════════
# Two views per sample:
#   weak   → random flip + mild colour jitter
#   strong → rotation + colour jitter + JPEG re-encode
# Both views share the SAME label → used for consistency regularisation.

class TwoViewDataset(Dataset):
    """Returns (img_weak, img_strong, label, domain_id) per sample."""
    def __init__(self, items): ...
    def __len__(self): ...
    def __getitem__(self, idx): ...
    # ... [implementation hidden]


class BalancedBatchSampler(Sampler):
    """
    Constructs batches with equal representation from each domain.
    Gracefully falls back to random batches when only one domain is present.
    """
    def __init__(self, items, batch_size, seed=42): ...
    def __len__(self): ...
    def __iter__(self): ...
    # ... [implementation hidden]


def make_loader(ds, sampler=None, shuffle=False) -> DataLoader:
    """Build DataLoader; supports both batch-level and index-level samplers."""
    # ... [implementation hidden]
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

# ── 5.1  Backbone Stems ──────────────────────────────────────────────────────

class ResNet50Stem(nn.Module):
    """ResNet-50 backbone → global avg pool → Linear(2048, proj_dim)."""
    def __init__(self, out_dim: int = 512): ...
    def forward(self, x): ...
    # ... [implementation hidden]


class ViTB16Stem(nn.Module):
    """ViT-B/16 backbone → CLS-token (or mean pool) → Linear(768, proj_dim)."""
    def __init__(self, out_dim: int = 512, img_size: int = 224): ...
    def forward(self, x): ...
    # ... [implementation hidden]


def build_stem(name: str, out_dim: int, img_size: int) -> nn.Module:
    """Factory — supported: 'resnet50' | 'vit_b16'."""
    # ... [implementation hidden]
    pass


# ── 5.2  Fusion Modules  (PRIMARY CONTRIBUTION) ──────────────────────────────
#
#  Given feature vectors a and b (proj_dim=512 each):
#
#  ConcatMLP           :  MLP( [a ‖ b] )
#  ConcatDeltaProd     :  MLP( [a ‖ b ‖ (a−b) ‖ (a⊙b)] )          ← PRIMARY
#  ConcatDeltaProdXAttn:  MLP( [a ‖ b ‖ (a−b) ‖ (a⊙b) ‖ xattn] )  ← extended
#
#  The delta term captures change magnitude; product term captures co-activation.

class ConcatMLP(nn.Module):
    """Baseline fusion: concat(a, b) → MLP."""
    # ... [implementation hidden]


class ConcatDeltaProd(nn.Module):
    """Proposed fusion: concat(a, b, a-b, a*b) → MLP.  [PRIMARY]"""
    # ... [implementation hidden]


class ConcatDeltaProdXAttn(nn.Module):
    """Extended fusion: ConcatDeltaProd + bidirectional cross-attention."""
    # ... [implementation hidden]


def build_fusion(name: str, feat_dim: int, out_dim: int) -> nn.Module:
    """Factory — supported: 'concat_mlp' | 'concat_delta_prod' | 'concat_delta_prod_xattn'."""
    # ... [implementation hidden]
    pass


# ── 5.3  Domain-FiLM Conditioning ────────────────────────────────────────────
#
#  z̃ = z · (1 + γ_d) + β_d
#  γ, β predicted from Embedding(num_domains, embed_dim) → small MLP

class DomainFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on domain ID."""
    def __init__(self, feat_dim: int, embed_dim: int = 64, num_domains: int = 2): ...
    def forward(self, feat, domain_ids): ...
    # ... [implementation hidden]


# ── 5.4  Full Model ───────────────────────────────────────────────────────────

class DualStemFusionNet(nn.Module):
    """
    Dual-backbone fusion network.

        stem_a(x_w), stem_b(x_w)  →  fuse  →  DomainFiLM  →  head   (weak view)
        stem_a(x_s), stem_b(x_s)  →  fuse  →  DomainFiLM  →  head   (strong view)

    Both logit streams are returned and combined via consistency KL loss.
    Supports progressive ViT-block / ResNet-stage unfreezing.
    """

    def __init__(
        self,
        stems=("vit_b16", "vit_b16"),
        fusion: str = "concat_mlp",
        proj_dim: int = 512,
        num_classes: int = 3,
        img_size: int = 224,
        domain_film_dim: int = 64,
        num_domains: int = 2,
        dropout: float = 0.2,
    ): ...

    def forward(self, x_w, x_s, domain_ids):
        """
        x_w        : weak-augmented image   (B, 3, H, W)
        x_s        : strong-augmented image (B, 3, H, W)
        domain_ids : domain labels          (B,)  long
        Returns    : logits_weak, logits_strong   each (B, num_classes)
        """
        # ... [implementation hidden]
        pass

    def freeze_backbones(self):
        """Freeze both stems; keep fusion, FiLM, dropout, head trainable."""
        # ... [implementation hidden]
        pass

    def unfreeze_one_stage(self):
        """Alternately unfreeze next ViT block or ResNet stage in stem_a / stem_b."""
        # ... [implementation hidden]
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
#
#  | Loss              | Formula                        | Used          |
#  |-------------------|-------------------------------|---------------|
#  | Cross-Entropy     | Standard CE                    | optional      |
#  | Label-Smoothing CE| CE with ε=0.1                  | optional      |
#  | Focal Loss        | (1−p_t)^γ · CE  (γ=2.0)       | ✅ primary    |
#  | Consistency KL    | KL(softmax_w ‖ softmax_s)      | ✅ λ=0.5      |
#
#  Total loss:  L = L_focal  +  0.5 × L_KL

class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.1): ...
    def forward(self, logits, target): ...
    # ... [implementation hidden]


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None): ...
    def forward(self, logits, target): ...
    # ... [implementation hidden]


def get_cls_loss(name: str) -> nn.Module:
    """Factory — supported: 'ce' | 'lsce' | 'focal'."""
    # ... [implementation hidden]
    pass


def consistency_kl(logits_w, logits_s) -> torch.Tensor:
    """KL divergence between weak- and strong-view predictions (consistency reg.)."""
    # ... [implementation hidden]
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 7. EMA & OPTIMISER GROUPS
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Exponential Moving Average of model weights.
    decay=0.999  |  use apply_to() / restore() around inference.
    """
    def __init__(self, model, decay: float = 0.999): ...
    def update(self, model): ...
    def apply_to(self, model): ...
    def restore(self, model): ...
    # ... [implementation hidden]


def build_optimizer(model) -> torch.optim.Optimizer:
    """
    Separate AdamW param groups:
      - ViT params    → lr × lr_vit_mult   (0.05)
      - ResNet params → lr × lr_resnet_mult (0.1)
      - Fusion / head → lr                  (1e-4)
    """
    # ... [implementation hidden]
    pass


def build_scheduler(optimizer, steps_per_epoch: int, epochs: int, stems=None):
    """
    Auto-selects scheduler:
      ViT + ViT  → OneCycleLR
      ResNet + * → CosineAnnealingLR
    CFG['scheduler']='plateau' overrides → ReduceLROnPlateau.
    """
    # ... [implementation hidden]
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 8. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
#
#  Key design decisions:
#    • AMP (autocast + GradScaler) throughout
#    • Gradient clipping at 1.0
#    • EMA update every step
#    • ReduceLROnPlateau stepped after validation (not per batch)
#    • Early stopping with patience=10; best EMA checkpoint auto-restored
#    • Optional conditional backbone unfreezing on plateau

def train_one_epoch(model, loader, optimizer, scaler, scheduler, ema, epoch, device) -> dict:
    """
    One training epoch with AMP + Focal loss + Consistency KL + EMA update.
    Returns: {'loss': float, 'acc': float}
    """
    # ... [implementation hidden]
    pass


@torch.no_grad()
def run_eval(model, loader, device, use_ema: bool = False, ema=None):
    """
    Evaluate model; optionally swap in EMA weights for inference.
    Returns: logits, y_true, probs, avg_loss
    """
    # ... [implementation hidden]
    pass


def fit(epochs: int, out_dir: str, tag: str = "full", stems=None, patience: int = 10):
    """
    Full training pipeline:
      1. Build DualStemFusionNet + freeze backbones
      2. Build per-group optimizers + schedulers
      3. Train with early stopping on val accuracy
      4. Save best / best-EMA / last checkpoints
    Returns: (best_path, best_ema_path, last_path)
    """
    # ... [implementation hidden]
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 9. CALIBRATION & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
#
#  Post-training calibration via temperature scaling (LBFGS on val set).
#
#  Metrics reported:
#    Accuracy · Macro-F1 · Balanced Accuracy
#    Quadratic Weighted Kappa (QWK) · Kendall τ
#    ROC-AUC (OvR macro) · PR-AUC (macro)
#    Log-loss · Brier score
#    ECE & MCE (15-bin reliability diagram)
#
#  Plots: Confusion matrix (raw + normalised) · ROC · PR · Reliability diagram

class ModelWithTemperature(nn.Module):
    """Wraps any model and divides logits by a learnable scalar temperature."""
    def __init__(self, model): ...
    def forward(self, x_w, x_s, d): ...
    # ... [implementation hidden]


def fit_temperature(model, val_loader, device) -> ModelWithTemperature:
    """Optimise temperature on validation set with LBFGS (NLL objective)."""
    # ... [implementation hidden]
    pass


def metric_report(logits, y, probs, out_prefix: str) -> tuple:
    """Compute all metrics and save JSON report. Returns (report_dict, confusion_matrix)."""
    # ... [implementation hidden]
    pass


def evaluate_checkpoint(ckpt_path: str, out_dir: str, tag: str = "full"):
    """Load checkpoint → fit temperature → evaluate val+test → save all plots."""
    # ... [implementation hidden]
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 10. RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
#
#  ┌─────────┬──────────┬──────────┬───────────────┬─────┐
#  │ Split   │ Accuracy │ Macro-F1 │ Balanced Acc  │ QWK │
#  ├─────────┼──────────┼──────────┼───────────────┼─────┤
#  │ Val     │  ~0.899  │    —     │      —        │  —  │
#  │ Test    │    —     │    —     │      —        │  —  │
#  └─────────┴──────────┴──────────┴───────────────┴─────┘
#
#  Best checkpoint: Epoch 38 / 50  |  Val acc 0.899
#  Config: fusion=concat_delta_prod, loss=focal(γ=2), consistency_λ=0.5, ema=0.999


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_dir = CFG["out_dir_root"]
    best_ckpt, best_ema_ckpt, last_ckpt = fit(
        epochs=CFG["epochs_full"],
        out_dir=out_dir,
        tag="full",
        patience=10,
    )
    evaluate_checkpoint(best_ema_ckpt, out_dir=out_dir, tag="full")