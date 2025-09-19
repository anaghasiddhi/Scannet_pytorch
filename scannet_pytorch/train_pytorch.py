import os, sys
import warnings
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*reflected list.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*amp.*")
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys, argparse, warnings
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import math
from time import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# === ScanNet imports ===
from scannet_pytorch.network.scannet import ScanNet
from scannet_pytorch.preprocessing.pipelines import ScanNetPipeline
from scannet_pytorch.utilities import paths
from scannet_pytorch.utilities.train_utils import (
    process_inputs, EarlyStopping, FocalLoss
)
from scannet_pytorch.utilities.data_loaders import get_train_loader, get_val_loader, get_test_loader
from scannet_pytorch.utilities.io_utils import save_labels_probs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", message=".*reflected list.*")


# === Helpers for DDP ===
def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_rank0():
    return get_rank() == 0

def ddp_setup(backend="nccl", timeout_sec=1800):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=timeout_sec)
    )
    return local_rank, dist.get_world_size()

def set_seed(seed: int = 1337):
    import os, random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- Batch helpers: keep structure + move to device ----------------
def _to_device(x, device):
    """Recursively move tensors (and containers of tensors) to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(e, device) for e in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x  # leave non-tensors (ints/None/strings) as-is

def _as_float_targets(t):
    if isinstance(t, torch.Tensor):
        t = t.float()
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim >= 2 and t.size(-1) == 1:
            t = t.squeeze(-1)
        if t.ndim >= 2 and t.size(-1) == 2:
            t = t[..., 1]  # take foreground
        # 🚑 force into [0,1]
        t = t.clamp(0, 1)
        return t
    if isinstance(t, (list, tuple)):
        return torch.tensor(t, dtype=torch.float32).clamp(0, 1)
    raise TypeError(f"Unsupported targets type: {type(t)}")

def _unpack_batch(batch, device):
    """
    Accepts whatever your DataLoader yields.
    Expects (inputs, targets) OR {"inputs": ..., "targets": ...}.
    Keeps inputs as dict/tuple/list/tensor and just moves tensors to device.
    """
    # Common cases: tuple/list of (inputs, targets)
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        inputs, targets = batch
    # Dict case: keys like {"inputs": ..., "targets": ...}
    elif isinstance(batch, dict) and "targets" in batch:
        inputs, targets = batch.get("inputs", batch), batch["targets"]
    else:
        raise ValueError(
            f"Unrecognized batch structure (type={type(batch)}). "
            "Expected (inputs, targets) or dict with 'targets'."
        )

    inputs = _to_device(inputs, device)
    if isinstance(targets, torch.Tensor):
        targets = targets.to(device, non_blocking=True)
    targets = _as_float_targets(targets)
    return inputs, targets



# ======== Eval helpers: operating points, neighbor cleanup, top-k =========
import numpy as _np
from sklearn.metrics import precision_recall_curve as _prc

def _r_at_p_threshold(y_true, y_prob, p_min=0.20):
    p, r, thr = _prc(y_true, y_prob)
    ok = _np.where(p >= p_min)[0]
    if len(ok) == 0:
        return None, 0.0  # no threshold attains that precision
    i = ok[_np.argmax(r[ok])]
    chosen_thr = float(thr[max(i-1, 0)]) if i < len(thr) else 0.5
    return chosen_thr, float(r[i])

def _p_at_r_threshold(y_true, y_prob, r_min=0.60):
    p, r, thr = _prc(y_true, y_prob)
    ok = _np.where(r >= r_min)[0]
    if len(ok) == 0:
        return None, 0.0
    i = ok[_np.argmax(p[ok])]
    chosen_thr = float(thr[max(i-1, 0)]) if i < len(thr) else 0.5
    return chosen_thr, float(p[i])

def _flatten_scalars(d, prefix=""):
    import numpy as _np
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from _flatten_scalars(v, name)
        else:
            # keep only plain numbers
            if isinstance(v, (int, float)) or (hasattr(v, "item") and callable(v.item)):
                try:
                    yield name, float(v)
                except Exception:
                    pass
def _per_chain_topk_metrics(batch_probs, batch_labels, ks=(5,)):
    """
    Compute per-chain precision/recall at top-k (e.g., 5).
    Returns dict: {"Prec@k", "Rec@k"} averaged over chains with >=1 positive.
    """
    import numpy as np
    out = {f"Prec@{k}": [] for k in ks}
    out.update({f"Rec@{k}": [] for k in ks})
    for probs, y in zip(batch_probs, batch_labels):
        probs = np.asarray(probs).reshape(-1)
        y = np.asarray(y).astype(np.uint8).reshape(-1)
        pos = int((y == 1).sum())
        if probs.size == 0:
            continue
        order = np.argsort(-probs)
        for k in ks:
            kk = min(k, probs.size)
            sel = order[:kk]
            tp = int((y[sel] == 1).sum())
            prec = tp / kk if kk > 0 else 0.0
            rec = tp / pos if pos > 0 else np.nan
            out[f"Prec@{k}"].append(prec)
            if not np.isnan(rec):
                out[f"Rec@{k}"].append(rec)
    return {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in out.items()}

import torch.nn.functional as F

def compute_weighted_bce_loss(logits, targets, pos_weight, use_hn=False,
                              top_q=0.10, hn_weight=2.0):
    """
    BCE-with-logits loss with pos_weight and optional hard-negative up-weighting.
    logits, targets: [B, L] or [L]
    """
    targets = targets.float()
    # element-wise BCE
    per_elem = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    if use_hn:
        with torch.no_grad():
            neg_mask = (targets == 0)
            if neg_mask.any():
                neg_logits = logits[neg_mask]
                qthr = torch.quantile(neg_logits.detach(), 1 - max(1e-6, min(0.999999, top_q)))
                hard_neg_mask = neg_mask & (logits >= qthr)
            else:
                hard_neg_mask = torch.zeros_like(targets, dtype=torch.bool)
        weights = torch.ones_like(per_elem)
        weights[hard_neg_mask] = float(hn_weight)
        per_elem = per_elem * weights
    return per_elem.mean()

# === Model ===
def init_model(device, dropout=0.25):
    return ScanNet(
        use_attention=True,
        nfeatures_aa=20,
        nfeatures_atom=12,
        nembedding_aa=16,
        nembedding_atom=12,
        nfilters_aa=64,
        nfilters_atom=16,
        nattentionheads_graph=1,
        nfilters_graph=2,
        activation='gelu',
        dropout=dropout,
        rotation_pooling=False,
        with_atom=True
    ).to(device)


# === Pipeline ===
pipeline = ScanNetPipeline(
    atom_features="valency",
    aa_features="sequence",
    aa_frames="triplet_sidechain",
    Beff=500
)
os.makedirs(paths.pipeline_folder, exist_ok=True)

# === ID loader ===
def load_ids(path):
    with open(path) as f:
        return [line.strip() for line in f]

# === Training loop (fixed: synchronized skips across ranks) ===
def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    epoch=0, log_every=50, scaler=None, accumulation_steps=1,
                    scheduler=None, args=None, ema=None, pos_weight=None):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    preds_buf, targs_buf = [], []

    for step, batch in enumerate(train_loader):
        inputs, targets = _unpack_batch(batch, device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            packed = process_inputs(inputs, device)
            if isinstance(packed, dict):
                outputs = model(**packed)
            elif isinstance(packed, (list, tuple)):
                outputs = model(*packed)
            else:
                outputs = model(packed)

            # ---- harmonize targets to match outputs [B, L] ----
            if isinstance(targets, (list, tuple)):
                targets = targets[0]
            targets = torch.as_tensor(targets, device=outputs.device, dtype=outputs.dtype)

            if outputs.ndim == targets.ndim + 1 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)

            if outputs.dim() == 2:
                B, L = outputs.shape
                if targets.dim() == 0:
                    targets = targets.view(1, 1).expand(B, L)
                elif targets.dim() == 1:
                    if targets.numel() == 1:
                        targets = targets.view(1, 1).expand(B, L)
                    elif targets.numel() == L:
                        targets = targets.unsqueeze(0)
                elif targets.dim() == 2:
                    if targets.shape == (1, 1):
                        targets = targets.expand(B, L)
                    elif targets.shape == (B, 1):
                        targets = targets.expand(B, L)
                    elif targets.shape == (1, L) and B > 1:
                        targets = targets.expand(B, L)

            if not torch.is_floating_point(targets):
                targets = targets.float()
            targets = targets.clamp(0, 1)
            if not torch.all((targets == 0) | (targets == 1)):
                targets = (targets > 0.5).float()

            # --- BCE with optional hard-negative up-weighting ---
            use_hn = getattr(args, "use_hard_negatives", False) and (epoch >= getattr(args, "hn_start_epoch", 1))

            # Choose a pos_weight safely
            pw = pos_weight
            if pw is None:
                if hasattr(criterion, "pos_weight") and getattr(criterion, "pos_weight") is not None:
                    pw = criterion.pos_weight
                elif getattr(args, "use_fixed_prevalence", False):
                    p = torch.tensor(getattr(args, "fixed_p", 0.0529), device=outputs.device, dtype=outputs.dtype)
                    p = torch.clamp(p, 1e-6, 1 - 1e-6)
                    pw = (1 - p) / p

            if isinstance(pw, torch.Tensor):
                pw = pw.to(device=outputs.device, dtype=outputs.dtype)

            loss = compute_weighted_bce_loss(
                outputs, targets,
                pos_weight=pw,
                use_hn=use_hn,
                top_q=getattr(args, "hn_top_q", 0.10),
                hn_weight=getattr(args, "hn_weight", 2.0)
            )
            # normalize for accumulation
            loss = loss / max(1, accumulation_steps)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                try:
                    ema.update()
                except Exception:
                    pass

        if not torch.isfinite(loss):
            print(f"[Train][Batch {step}] ❌ Non-finite loss detected: {loss.item()}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            preds_buf.clear(); targs_buf.clear()
            continue

        total_loss += float(loss.detach().item())

        # collect preds/targets for AUROC snapshot
        with torch.no_grad():
            probs = torch.sigmoid(outputs).detach().float().view(-1).cpu().numpy()
            t_bin = targets.detach().float().clamp(0, 1).view(-1).cpu().numpy()
            preds_buf.append(probs)
            targs_buf.append(t_bin)

        # log progress
        if ((step + 1) % log_every == 0) or ((step + 1) == num_batches):
            try:
                import numpy as np
                from sklearn.metrics import roc_auc_score
                preds = np.concatenate(preds_buf) if preds_buf else np.array([])
                targs = np.concatenate(targs_buf) if targs_buf else np.array([])
                if preds.size > 0 and targs.size == preds.size:
                    has_pos = (targs == 1).any()
                    has_neg = (targs == 0).any()
                    if has_pos and has_neg:
                        auroc = roc_auc_score(targs, preds)
                        print(f"[Train] {step+1}/{num_batches} | loss={loss.item():.4f} | AUROC={auroc:.4f}", flush=True)
                    else:
                        print(f"[Train] {step+1}/{num_batches} | loss={loss.item():.4f} | AUROC=N/A", flush=True)
                else:
                    print(f"[Train] {step+1}/{num_batches} | loss={loss.item():.4f}", flush=True)
            finally:
                preds_buf.clear()
                targs_buf.clear()

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    return avg_loss, None

# === Validation (Phase-0 metrics) ===
@torch.no_grad()
def validate(model, dataloader, criterion, device, log_dir="logs",
             run_name="test_runs", split_name="val", epoch=None):
    model.eval()
    val_losses = []

    all_probs_flat, all_targets_flat = [], []
    topk_per_chain_metrics = {"Prec@5": [], "Rec@5": []}  # only Top-5%

    for step, batch in enumerate(dataloader):
        try:
            inputs, t = _unpack_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=False):
                packed = process_inputs(inputs, device)
                logits = model(**packed) if isinstance(packed, dict) \
                    else model(*packed) if isinstance(packed, (list, tuple)) \
                    else model(packed)

                # --- harmonize targets to match logits [B, L] ---
                if isinstance(t, (list, tuple)):
                    t = t[0]
                t = torch.as_tensor(t, device=logits.device, dtype=logits.dtype)

                if logits.ndim == t.ndim + 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)

                if logits.dim() == 2:
                    B, L = logits.shape
                    if t.dim() == 0:
                        t = t.view(1, 1).expand(B, L)
                    elif t.dim() == 1:
                        if t.numel() == 1:
                            t = t.view(1, 1).expand(B, L)
                        elif t.numel() == L:
                            t = t.unsqueeze(0)
                    elif t.dim() == 2:
                        if t.shape == (1, 1):
                            t = t.expand(B, L)
                        elif t.shape == (B, 1):
                            t = t.expand(B, L)
                        elif t.shape == (1, L) and B > 1:
                            t = t.expand(B, L)

                if not torch.is_floating_point(t):
                    t = t.float()
                t = t.clamp(0, 1)
                if not torch.all((t == 0) | (t == 1)):
                    t = (t > 0.5).float()

                if logits.shape != t.shape:
                    if is_rank0():
                        print(f"⚠️ Validation batch harmonization failed: preds={tuple(logits.shape)} targets={tuple(t.shape)}")
                    continue

                loss_val = float(criterion(logits, t).item())

            if not math.isfinite(loss_val):
                if is_rank0():
                    print("⚠️ Skipped a non-finite validation loss.")
                continue
            val_losses.append(loss_val)

            # ---- collect flat for global metrics ----
            probs = torch.sigmoid(logits).detach().float()
            t_bin = _as_float_targets(t).detach().float().clamp(0, 1)

            p_flat = probs.view(-1).cpu().numpy()
            y_flat = t_bin.view(-1).cpu().numpy()
            if p_flat.size != y_flat.size:
                m = min(p_flat.size, y_flat.size)
                if m == 0:
                    continue
                p_flat, y_flat = p_flat[:m], y_flat[:m]

            all_probs_flat.append(p_flat)
            all_targets_flat.append(y_flat)

            # ---- per-chain Top-5% (robust to degenerate batches) ----
            probs_np, t_np = probs.cpu().numpy(), t_bin.cpu().numpy()
            try:
                if probs_np.ndim == 2 and probs_np.shape[0] > 1:
                    batch_probs = [probs_np[b] for b in range(probs_np.shape[0])]
                    batch_labels = [t_np[b] for b in range(t_np.shape[0])]
                else:
                    batch_probs = [probs_np.reshape(-1)]
                    batch_labels = [t_np.reshape(-1)]
                mks = _per_chain_topk_metrics(batch_probs, batch_labels, ks=(5,))
                for k in mks:
                    topk_per_chain_metrics[k].append(mks[k])
            except Exception as e:
                if is_rank0():
                    print(f"⚠️ Skipping Top-K metrics for batch {step}: {e}")
                pass

        except Exception as e:
            if is_rank0():
                print(f"⚠️ Validation batch skipped due to error: {e}")
            continue

    # ===== end of dataloader loop =====
    if not all_probs_flat or not all_targets_flat:
        return None

    all_preds = np.concatenate(all_probs_flat)
    all_targets = np.concatenate(all_targets_flat)
    if all_preds.size != all_targets.size:
        m = min(all_preds.size, all_targets.size)
        all_preds, all_targets = all_preds[:m], all_targets[:m]

    # save per-example predictions for threshold sweeping
    save_labels_probs(
        os.path.join(log_dir, f"ops_{split_name}.examples.json"),
        all_targets,
        all_preds,
        are_logits=False
    )

    # ---- global metrics ----
    has_pos = (all_targets == 1).any()
    has_neg = (all_targets == 0).any()

    try:
        auroc = roc_auc_score(all_targets, all_preds) if has_pos and has_neg else 0.0
    except Exception:
        auroc = 0.0

    try:
        pr, rc, _ = precision_recall_curve(all_targets, all_preds)
        pr_auc = auc(rc, pr)
    except Exception:
        pr_auc, pr, rc = 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])

    p_base = float(all_targets.mean()) if all_targets.size > 0 else 0.0
    lift = (pr_auc / p_base) if p_base > 0 else 0.0
    norm_auprc = ((pr_auc - p_base) / (1 - p_base)) if p_base < 1.0 else 0.0

    pred_labels = (all_preds >= 0.5).astype(int)
    try:
        f1 = f1_score(all_targets, pred_labels, zero_division=0)
        precision = precision_score(all_targets, pred_labels, zero_division=0)
        recall = recall_score(all_targets, pred_labels, zero_division=0)
        acc = accuracy_score(all_targets, pred_labels)
    except Exception:
        f1 = precision = recall = acc = 0.0

    # ---- P@R=0.50 ----
    thr_r50, p_at_r50 = _p_at_r_threshold(all_targets, all_preds, r_min=0.50)
    def _metrics_at_thr(probs, y, thr):
        yhat = (probs >= thr).astype(int)
        return {
            "Precision": float(precision_score(y, yhat, zero_division=0)),
            "Recall":    float(recall_score(y, yhat, zero_division=0)),
            "F1":        float(f1_score(y, yhat, zero_division=0)),
        }
    op = {}
    if thr_r50 is not None:
        op["P_at_R=0.50"] = {"thr": float(thr_r50), **_metrics_at_thr(all_preds, all_targets, thr_r50)}
    else:
        op["P_at_R=0.50"] = {"thr": None, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    # ---- aggregate Top-5% ----
    topk_final = {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in topk_per_chain_metrics.items()}

    metrics = {
        "AUROC": auroc,
        "PR_AUC": pr_auc,
        "BaseRate_p": p_base,
        "Lift": lift,
        "NormAUPRC": norm_auprc,
        "F1": f1, "Precision": precision, "Recall": recall,
        "Accuracy": acc,
        "Loss": float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0,
        "P_at_R=0.50": op["P_at_R=0.50"],
        "TopK": topk_final,
    }

    if is_rank0():
        os.makedirs(log_dir, exist_ok=True)
        try:
            plt.figure()
            plt.plot(rc, pr)
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title(f"{split_name} PR Curve (AUC={pr_auc:.4f})")
            plt.savefig(os.path.join(log_dir, f"pr_curve_{split_name}.png"))
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not save PR curve: {e}")

        try:
            import json
            with open(os.path.join(log_dir, f"ops_{split_name}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save ops json: {e}")

        ep_str = f"(epoch {epoch}) " if epoch is not None else ""
        print(f"🔎 [{split_name}] Metrics {ep_str}")
        print(f"  • AUROC={auroc:.4f}  PR_AUC={pr_auc:.4f}  BaseRate={p_base:.4f}  Lift={lift:.3f}  NormAUPRC={norm_auprc:.3f}")
        d = metrics['P_at_R=0.50']
        print(f"  • P@R=0.50: thr={d['thr']}  P={d['Precision']:.3f} R={d['Recall']:.3f} F1={d['F1']:.3f}")
        print(f"  • Top-5% per chain: Prec={metrics['TopK']['Prec@5']:.3f}, Rec={metrics['TopK']['Rec@5']:.3f}")

    return metrics


class ExponentialMovingAverage:
    def __init__(self, params, decay=0.999):
        self.decay = decay
        self.params = [p for p in params if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    def update(self):
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                s.mul_(self.decay).add_(p, alpha=1 - self.decay)

    from contextlib import contextmanager
    @contextmanager
    def average_parameters(self):
        backup = [p.detach().clone() for p in self.params]
        try:
            for p, s in zip(self.params, self.shadow):
                p.data.copy_(s.data)
            yield
        finally:
            for p, b in zip(self.params, backup):
                p.data.copy_(b.data)

# === Main ===
def main():
    import argparse, os, torch
    from time import time
    from torch.utils.tensorboard import SummaryWriter
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    parser = argparse.ArgumentParser()

    # Training basics
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    # Paths & logging
    parser.add_argument("--log_dir", type=str, default="logs/test_runs")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--use_processed_ids", action="store_true")
    parser.add_argument("--train_ids", type=str, default="data/train_processed_ids.txt")
    parser.add_argument("--val_ids", type=str, default="data/val_processed_ids.txt")
    parser.add_argument("--test_ids", type=str, default="data/test_processed_ids.txt")

    # Optimizer & LR
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["cosine", "onecycle", "plateau"])

    # Model & regularization
    parser.add_argument("--dropout", type=float, default=0.25)

    # Loss
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # Sparse-label loss tweak
    parser.add_argument("--use_fixed_prevalence", action="store_true",
                        help="Use fixed split prevalence to set BCE pos_weight=(1-p)/p")
    parser.add_argument("--fixed_p", type=float, default=0.0529,
                        help="Fixed prevalence p for pos_weight when --use_fixed_prevalence is set")

    # EMA
    parser.add_argument("--use_ema", action="store_true",
                        help="Track an exponential moving average of weights for validation")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Hard Negative Mining
    parser.add_argument("--use_hard_negatives", action="store_true")
    parser.add_argument("--hn_start_epoch", type=int, default=1)
    parser.add_argument("--hn_top_q", type=float, default=0.10)
    parser.add_argument("--hn_weight", type=float, default=2.0)


    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (e.g. best_model_val.pt)")

    args = parser.parse_args()
    set_seed(args.seed)

    # ---------------- DDP setup ----------------
    if args.dist:
        ddp_setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Logging ----------------
    if is_rank0():
        os.makedirs(args.log_dir, exist_ok=True)
        print(f"📝 TensorBoard logs → {args.log_dir}")
        print(f"🖥️ Training on {device.type} | world_size={'N/A' if not is_dist() else dist.get_world_size()}")

    writer = SummaryWriter(log_dir=args.log_dir) if is_rank0() else None

    # ---------------- Model & pipeline ----------------
    # pipeline is already created at module scope: ScanNetPipeline(...)
    model = init_model(device, dropout=args.dropout)

    if args.dist:
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
                    output_device=int(os.environ.get("LOCAL_RANK", 0)),
                    find_unused_parameters=False)

    # ---------------- ID loading ----------------
    train_ids = load_ids(args.train_ids) if args.use_processed_ids else None
    val_ids   = load_ids(args.val_ids)   if args.use_processed_ids else None
    test_ids  = load_ids(args.test_ids)  if args.use_processed_ids else None

    # ---------------- Data loaders ----------------
    train_loader = get_train_loader(
        pipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_ids=train_ids
    )
    val_loader = get_val_loader(
        pipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_ids=val_ids
    )
    test_loader = get_test_loader(
        pipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_ids=test_ids
    )

    val_loaders = {"val": val_loader, "test": test_loader}

    # ---------------- Samplers for DDP (if needed) ----------------
    train_sampler = None
    if args.dist:
        if hasattr(train_loader, "sampler") and not isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader.dataset, shuffle=True)
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
        if hasattr(val_loader, "sampler"):
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_loader.dataset, shuffle=False)
            val_loader = torch.utils.data.DataLoader(
                val_loader.dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
        if hasattr(test_loader, "sampler"):
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_loader.dataset, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                test_loader.dataset,
                batch_size=args.batch_size,
                sampler=test_sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )

    # ---------------- Loss setup (supports fixed prevalence) ----------------
    if args.use_fixed_prevalence:
        p = max(1e-6, min(1 - 1e-6, args.fixed_p))
        pos_weight = torch.tensor([(1 - p) / p], device=device)
    else:
        pos, neg = 0, 0
        sample_batches = 5
        for bi, (_, targets) in enumerate(train_loader):
            t = torch.as_tensor(targets).view(-1).cpu()
            pos += int((t == 1).sum())
            neg += int((t == 0).sum())
            if bi + 1 >= sample_batches:
                break
        if pos == 0:
            pos_weight = torch.tensor([1.0], device=device)
        else:
            scale = float(os.environ.get("POS_WEIGHT_SCALE", "1.0"))
            pos_weight = torch.tensor([scale * (neg / max(1, pos))], device=device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    if is_rank0():
        try:
            _pw = pos_weight.item() if isinstance(pos_weight, torch.Tensor) else float(pos_weight)
            print(f"🔧 Using pos_weight={_pw:.4f}")
        except Exception:
            print(f"🔧 Using pos_weight={pos_weight}")

    # ---------------- Warm-up forward ----------------
    model.train()
    try:
        warmup_batch = next(iter(train_loader))
        warmup_inputs, _ = _unpack_batch(warmup_batch, device)
        packed = process_inputs(warmup_inputs, device)
        with torch.no_grad():
            if isinstance(packed, dict):
                _ = model(**packed)
            elif isinstance(packed, (list, tuple)):
                _ = model(*packed)
            else:
                _ = model(packed)
        if is_rank0():
            print("✅ Warm-up forward completed; LazyLinear initialized.")
        torch.cuda.empty_cache()
    except Exception as e:
        if is_rank0():
            print(f"⚠️ Warm-up forward failed: {e}")

    # ---------------- Optimizer / Scaler / Scheduler ----------------
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.1, div_factor=25.0, final_div_factor=1e4
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay) if args.use_ema else None
    best_aurocs = {split: 0.0 for split in val_loaders}
    early_stopper = EarlyStopping(patience=5)

    # ---------------- CSV metrics log ----------------
    csv_log_path = os.path.join(args.log_dir, "metrics_log.csv")
    if is_rank0() and not os.path.exists(csv_log_path):
        os.makedirs(args.log_dir, exist_ok=True)
        with open(csv_log_path, "w") as f:
            f.write("Epoch,Split,AUROC,PR_AUC,BaseRate,Lift,NormAUPRC,F1,Precision,Recall,Accuracy,Loss,"
                    "P_atR50_Precision,P_atR50_Recall\n")

    # ---------------- Eval-only shortcut ----------------
    if getattr(args, "eval_only", False):
        if args.resume:
            print(f"🔄 Loading checkpoint from {args.resume}")
            state_dict = torch.load(args.resume, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            print("⚠️ No --resume path provided, evaluating with randomly initialized weights.")

        for split_name, vloader in val_loaders.items():
            validate(
                model.module if args.dist else model,
                vloader, criterion, device,
                log_dir=args.log_dir,
                run_name=os.path.basename(args.log_dir),
                split_name=split_name,
                epoch=0
            )
        return
    # -----------------------------------------------------


    for epoch in range(args.epochs):
        if args.dist and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_rank0():
            print(f"\n🌀 Epoch {epoch+1}/{args.epochs}")
        start_time = time()

        train_loss, _ = train_one_epoch(
            model.module if args.dist else model,
            train_loader,
            optimizer,
            criterion,
            device,
            accumulation_steps=args.accumulation_steps,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            ema=ema,
            pos_weight=pos_weight
        )

        if is_rank0() and writer:
            writer.add_scalar("Train/Loss", train_loss, epoch)

        val_metrics_by_split = {}
        current_val = 0.0
        if is_rank0():
            for split_name, vloader in val_loaders.items():
                if args.use_ema and ema is not None:
                    with ema.average_parameters():
                        val_metrics = validate(
                            model.module if args.dist else model,
                            vloader, criterion, device,
                            log_dir=args.log_dir,
                            run_name=os.path.basename(args.log_dir),
                            split_name=split_name,
                            epoch=epoch
                        )
                else:
                    val_metrics = validate(
                        model.module if args.dist else model,
                        vloader, criterion, device,
                        log_dir=args.log_dir,
                        run_name=os.path.basename(args.log_dir),
                        split_name=split_name,
                        epoch=epoch
                    )

                if val_metrics:
                    val_metrics_by_split[split_name] = val_metrics
                    print(f"📊 {split_name} — AUROC={val_metrics['AUROC']:.4f}, "
                          f"PR_AUC={val_metrics['PR_AUC']:.4f}, Lift={val_metrics['Lift']:.3f}, "
                          f"P@R=0.50 P={val_metrics['P_at_R=0.50']['Precision']:.3f}")

                    if writer:
                        for name, value in _flatten_scalars(val_metrics, prefix=split_name):
                            writer.add_scalar(name, value, epoch)

                    if val_metrics["PR_AUC"] > best_aurocs.get(split_name, 0.0):
                        best_aurocs[split_name] = val_metrics["PR_AUC"]
                        os.makedirs(args.save_dir, exist_ok=True)
                        ckpt_path = os.path.join(args.save_dir, f"best_model_{split_name}.pt")
                        torch.save(
                            model.module.state_dict() if args.dist else model.state_dict(),
                            ckpt_path
                        )
                        print(f"✅ Saved new best model for {split_name}: {ckpt_path}")

            current_val = val_metrics_by_split.get("val", {}).get("PR_AUC", 0.0)

        # ---- Scheduler stepping ----
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(current_val)
            elif args.scheduler == "cosine":
                scheduler.step()

        stop_now = early_stopper.step(current_val)

        if args.dist:
            flag = torch.tensor(1 if stop_now else 0, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            stop_now = (flag.item() > 0)

        if is_rank0():
            with open(csv_log_path, "a") as f:
                for split, m in val_metrics_by_split.items():
                    f.write(f"{epoch},{split},{m['AUROC']:.4f},{m['PR_AUC']:.4f},{m['BaseRate_p']:.4f},"
                            f"{m['Lift']:.3f},{m['NormAUPRC']:.3f},{m['F1']:.4f},{m['Precision']:.4f},"
                            f"{m['Recall']:.4f},{m['Accuracy']:.4f},{m['Loss']:.4f},"
                            f"{m['P_at_R=0.50']['Precision']:.4f},{m['P_at_R=0.50']['Recall']:.4f}\n")
            print(f"🕒 Epoch time: {(time()-start_time)/60:.2f} min")

        if stop_now:
            if is_rank0():
                print(f"⏹ Early stopping at epoch {epoch+1}")
            break

    if is_rank0():
        if writer:
            writer.close()
        print("\n🏁 Training complete.")
        for split, score in best_aurocs.items():
            print(f"🔹 {split:<10} → {score:.4f}")

    if args.dist and is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
