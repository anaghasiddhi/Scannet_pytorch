#!/usr/bin/env python3
import json, math, argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def _to_np(a, dtype=float):
    return np.asarray(list(a), dtype=dtype)

def _maybe_load_sibling_labels(path: Path):
    # try ops_val.labels.json next to ops_val.json
    cand = path.with_suffix(path.suffix.replace(".json", "")).with_name(path.stem + ".labels.json")
    # if above doesn't exist, try "<name>.labels.json" exactly
    if not cand.exists():
        cand = path.parent / (path.stem + ".labels.json")
    if cand.exists():
        with open(cand, "r") as f:
            arr = json.load(f)
        return _to_np(arr, dtype=int)
    return None

def load_ops(path: Path):
    """
    Returns (y_true, y_prob) as np.ndarrays in [0,1].
    Handles these shapes:
      1) {"labels":[...], "probs":[...]}  or {"labels":[...], "logits":[...]}
      2) {"y_true":[...], "y_prob":[...]} or {"targets":[...], "scores":[...]}
      3) [{"label":..,"prob":..}, ...]    or [{"label":..,"logit":..}, ...]
      4) [[label, prob], ...]             or [[label, logit], ...]
      5) [prob, prob, ...]  + sibling file <path>.labels.json with labels
    """
    with open(path, "r") as f:
        obj = json.load(f)

    # Case 1/2: dict with arrays
    if isinstance(obj, dict):
        # normalize keys
        keys = {k.lower(): k for k in obj.keys()}
        lbl_key = None
        prob_key = None
        logit_key = None
        for cand in ["labels", "y_true", "targets", "gt"]:
            if cand in keys:
                lbl_key = keys[cand]; break
        for cand in ["probs", "prob", "y_prob", "scores"]:
            if cand in keys:
                prob_key = keys[cand]; break
        for cand in ["logits", "logit"]:
            if cand in keys:
                logit_key = keys[cand]; break

        if lbl_key is not None and (prob_key is not None or logit_key is not None):
            y_true = _to_np(obj[lbl_key], dtype=int)
            if prob_key is not None:
                y_prob = _to_np(obj[prob_key], dtype=float)
            else:
                y_prob = sigmoid(_to_np(obj[logit_key], dtype=float))
            if y_true.shape[0] != y_prob.shape[0]:
                raise ValueError(f"Length mismatch: labels {y_true.shape[0]} vs probs {y_prob.shape[0]}")
            return y_true, y_prob

        # dict of dicts (id -> {...})
        if all(isinstance(v, dict) for v in obj.values()):
            records = list(obj.values())
            # fallthrough to list-of-dicts handling
            obj = records
        else:
            # show a short diagnostic
            preview = {k: type(v).__name__ for k, v in list(obj.items())[:5]}
            raise ValueError(f"Unrecognized dict format in {path}. Keys→types: {preview}")

    # Case 3: list of dicts
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        y_true, y_prob = [], []
        for r in obj:
            # labels
            label = (r.get("label") if "label" in r else
                     r.get("y_true") if "y_true" in r else
                     r.get("target") if "target" in r else
                     r.get("gt") if "gt" in r else None)
            if label is None:
                raise ValueError("List-of-dicts format but no label key found (label/y_true/target/gt).")
            # score
            if "prob" in r:
                p = float(r["prob"])
            elif "probs" in r:
                p = float(r["probs"])
            elif "score" in r:
                p = float(r["score"])
            elif "y_prob" in r:
                p = float(r["y_prob"])
            elif "logit" in r:
                p = sigmoid(float(r["logit"]))
            elif "logits" in r:
                p = sigmoid(float(r["logits"]))
            else:
                raise ValueError("List-of-dicts format but no prob/score/logit key found.")
            y_true.append(int(label)); y_prob.append(p)
        return _to_np(y_true, int), _to_np(y_prob, float)

    # Case 4: list of pairs
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (list, tuple)):
        if len(obj[0]) < 2:
            raise ValueError("List-of-lists found but pairs do not have 2+ elements.")
        pairs = _to_np(obj, float)
        y_true = pairs[:, 0].astype(int)
        raw    = pairs[:, 1].astype(float)
        # Heuristic: if values go outside [0,1], treat as logits
        if (raw < 0).any() or (raw > 1).any():
            y_prob = sigmoid(raw)
        else:
            y_prob = raw
        return y_true, y_prob

    # Case 5: list of floats (probs) + sibling labels file
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (float, int)):
        probs = _to_np(obj, float)
        labels = _maybe_load_sibling_labels(path)
        if labels is None:
            raise ValueError(
                f"{path.name} looks like a list of probabilities (len={len(probs)}), "
                f"but I couldn’t find a sibling labels file '{path.stem}.labels.json'. "
                f"Create one (list of 0/1 of same length) or export a dict with arrays: "
                f'{{"labels":[...], "probs":[...]}}'
            )
        if labels.shape[0] != probs.shape[0]:
            raise ValueError(f"Length mismatch: labels {labels.shape[0]} vs probs {probs.shape[0]}")
        return labels, probs

    # Final fallback: show a tiny preview
    preview = str(obj)[:200]
    raise ValueError(f"Unrecognized JSON structure in {path}. Preview: {preview}")

def f_beta(prec, rec, beta=1.0):
    # Elementwise F_beta from P,R arrays
    beta2 = beta * beta
    num = (1 + beta2) * (prec * rec)
    den = (beta2 * prec) + rec
    f = np.where(den > 0, num / den, 0.0)
    return f

def evaluate_at_threshold(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(np.int32)
    tp = np.sum((y_hat == 1) & (y_true == 1))
    fp = np.sum((y_hat == 1) & (y_true == 0))
    fn = np.sum((y_hat == 0) & (y_true == 1))
    tn = np.sum((y_hat == 0) & (y_true == 0))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = f1_score(y_true, y_hat, zero_division=0)
    f05  = f_beta(np.array([prec]), np.array([rec]), beta=0.5)[0]
    acc  = (tp + tn) / max(1, len(y_true))
    return dict(threshold=thr, precision=prec, recall=rec, f1=f1, f0_5=f05, accuracy=acc,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--test", required=False, type=Path)
    ap.add_argument("--target_precision", type=float, default=0.20)
    ap.add_argument("--prefer", choices=["f0_5","f1","target_precision"], default="f0_5",
                    help="Selection rule for the primary operating point.")
    args = ap.parse_args()

    yv, pv = load_ops(args.val)
    prec, rec, thr = precision_recall_curve(yv, pv)
    # sklearn returns thresholds of length N-1; align arrays by trimming first P/R element
    thr = np.concatenate([np.array([1.0]), thr])  # prepend 1 so arrays match lengths
    assert len(prec) == len(rec) == len(thr)

    # Compute F-scores across curve
    f1_arr  = f_beta(prec, rec, beta=1.0)
    f05_arr = f_beta(prec, rec, beta=0.5)

    # Candidates
    idx_f05 = int(np.nanargmax(f05_arr))
    idx_f1  = int(np.nanargmax(f1_arr))

    # Target-precision candidate: pick max recall among points with P >= target
    mask = prec >= args.target_precision
    if mask.any():
        idx_tar = int(np.argmax(rec * mask))  # highest recall among qualified
    else:
        idx_tar = idx_f05  # fallback

    pick_map = {"f0_5": idx_f05, "f1": idx_f1, "target_precision": idx_tar}
    idx = pick_map[args.prefer]

    # Evaluate chosen threshold on val
    chosen_thr = float(thr[idx])
    val_stats  = evaluate_at_threshold(yv, pv, chosen_thr)

    # Also compute the target-precision operating point (useful to print)
    tar_thr = float(thr[idx_tar])
    val_tar = evaluate_at_threshold(yv, pv, tar_thr)

    # Print summary
    def row(d):
        return (f"thr={d['threshold']:.4f} | P={d['precision']:.3f} "
                f"R={d['recall']:.3f} F1={d['f1']:.3f} F0.5={d['f0_5']:.3f} Acc={d['accuracy']:.3f} "
                f"| TP={d['tp']} FP={d['fp']} FN={d['fn']} TN={d['tn']}")
    print("\nValidation sweep:")
    print(f"  * Best F0.5 → {row(evaluate_at_threshold(yv, pv, float(thr[idx_f05])))}")
    print(f"  * Best F1   → {row(evaluate_at_threshold(yv, pv, float(thr[idx_f1])))}")
    if mask.any():
        print(f"  * Max Recall with P≥{args.target_precision:.2f} → {row(val_tar)}")
    else:
        print(f"  * No point with P≥{args.target_precision:.2f}; showing Best F0.5 instead.")

    print("\nChosen operating point (per --prefer):")
    print("  ", row(val_stats))

    # Evaluate on test if provided
    if args.test and args.test.exists():
        yt, pt = load_ops(args.test)
        test_stats = evaluate_at_threshold(yt, pt, chosen_thr)
        print("\nTest @ chosen threshold:")
        print("  ", row(test_stats))

        # also test the target-precision point
        if mask.any():
            test_tar = evaluate_at_threshold(yt, pt, tar_thr)
            print(f"\nTest @ P≥{args.target_precision:.2f} (val-chosen) threshold:")
            print("  ", row(test_tar))

    # Save chosen thresholds to a sidecar json
    out = {
        "chosen_by": args.prefer,
        "threshold": chosen_thr,
        "val_stats": val_stats,
        "target_precision": args.target_precision,
        "target_precision_threshold": tar_thr,
        "val_stats_target_precision": val_tar,
    }
    out_path = args.val.with_suffix(".thresholds.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved thresholds → {out_path}")

if __name__ == "__main__":
    main()
