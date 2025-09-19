import json, os
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, average_precision_score, roc_auc_score
import numpy as np

def load_labels_probs(path):
    """Robust loader: handles common key names."""
    data = json.load(open(path))
    # try common keys
    keys = {
        "y_true": ["labels","label","targets","y_true","y","t","gt"],
        "probs":  ["probs","probabilities","scores","preds","p","y_prob"]
    }
    def pick(d, cand):
        for k in cand:
            if k in d:
                return np.asarray(d[k])
        raise KeyError(f"None of {cand} found in {path}. Keys: {list(d.keys())}")
    y = pick(data, keys["y_true"]).astype(int)
    p = pick(data, keys["probs"]).astype(float)
    # trim to same length if needed
    m = min(len(y), len(p))
    return y[:m], p[:m]

def load_thr(path_json):
    """Get P@R=0.50 threshold if saved; else None."""
    if not os.path.exists(path_json): return None
    j = json.load(open(path_json))
    try:
        return float(j["P_at_R=0.50"]["thr"])
    except Exception:
        return None

def summarize(split_dir, split_name):
    examples = Path(split_dir) / f"ops_{split_name}.examples.json"
    summary  = Path(split_dir) / f"ops_{split_name}.json"
    if not examples.exists():
        print(f"❌ Missing {examples}")
        return
    y, p = load_labels_probs(examples)
    thr_pr50 = load_thr(summary)

    def report(thr, label):
        yhat = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        acc = accuracy_score(y, yhat)
        # For context (threshold-free):
        try:
            aupr = average_precision_score(y, p)
        except Exception:
            aupr = float("nan")
        try:
            auroc = roc_auc_score(y, p)
        except Exception:
            auroc = float("nan")
        print(f"\n[{split_name} @ {label}] thr={thr:.4f}")
        print(f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
        print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  Acc={acc:.4f}  AUCPR={aupr:.4f}  AUROC={auroc:.4f}")

    # Default 0.5
    report(0.5, "0.50 default")
    # P@R=0.50 threshold if available
    if thr_pr50 is not None:
        report(thr_pr50, "P@R=0.50")

# ==== EDIT THIS PATH to your run folder ====
run_dir = "logs/train/phase1_bce_longer_20250906_2327"
for split in ["val", "test"]:
    summarize(run_dir, split)
