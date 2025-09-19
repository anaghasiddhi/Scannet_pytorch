#!/usr/bin/env python3
"""
make_run_report.py  —  report generator for your train_pytorch.py

What it reads (all optional except --run_dir):
  - <run_dir>/metrics_log.csv  (written by your train script each epoch)
  - --log <path to nohup.out>  (to capture PR_AUC per epoch + epoch times + ckpt paths)
  - <run_dir>/ops_val.json and ops_test.json (last-epoch operating points & top-k)

What it writes into <run_dir>/report/:
  - metrics.csv            (epoch-wise AUROC/F1/etc + PR_AUC if found in log)
  - summary.json           (best epochs, lifts, ckpt paths, total time)
  - report.md              (human-friendly summary)
  - curves.png             (PR_AUC/AUROC/Loss per epoch if matplotlib available)

Usage:
  python make_run_report.py \
    --run_dir logs/train/phaseD_atoms_labels_enriched_20250905_1949 \
    --log    logs/train/phaseD_atoms_labels_enriched_20250905_1949/nohup.out \
    --base_rate 0.0529
"""
import argparse, csv, json, re, os
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

SPLITS = ("train","val","test")

# --- unicode/format tolerant patterns that match your prints ---
GE = r"(?:>=|\u2265)"           # >= or ≥
ARROW = r"(?:->|→|\u2192)"      # -> or → (unicode)
DASHES = r"[\u2012-\u2015\u2212\-]"  # various dashes including minus

EPOCH_HDR = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
METRIC_LINE = re.compile(
    rf"(?P<split>train|val|test).*?AUROC\s*[:=]\s*(?P<auroc>[0-9.]+)\s*,\s*F1\s*[:=]\s*(?P<f1>[0-9.]+)\s*,\s*PR{DASHES}?_?AUC\s*[:=]\s*(?P<prauc>[0-9.]+)",
    re.IGNORECASE,
)
FINAL_ARROW = re.compile(
    rf"^\s*[\W_]*\b(?P<split>train|val|test)\b[^\d\n]*{ARROW}\s*(?P<prauc>[0-9.]+)\s*$",
    re.IGNORECASE,
)
EPOCH_TIME = re.compile(r"Epoch time:\s*([0-9.]+)\s*min", re.IGNORECASE)
BEST_SAVE  = re.compile(r"Saved new best model for\s+(train|val|test)\s*:\s*(\S+)", re.IGNORECASE)

def safe_float(x):
    try: return float(x)
    except Exception: return None

def read_metrics_csv(run_dir: Path):
    """Read <run_dir>/metrics_log.csv written by train_pytorch.py."""
    path = run_dir / "metrics_log.csv"
    rows = []
    if path.exists():
        with path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    rows.append({
                        "epoch": int(row["Epoch"]),
                        "split": row["Split"].lower(),
                        "AUROC": safe_float(row.get("AUROC")),
                        "F1": safe_float(row.get("F1")),
                        "Precision": safe_float(row.get("Precision")),
                        "Recall": safe_float(row.get("Recall")),
                        "Accuracy": safe_float(row.get("Accuracy")),
                        "Loss": safe_float(row.get("Loss")),
                    })
                except Exception:
                    pass
    return rows

def parse_nohup(log_path: Path):
    """Parse nohup.out to attach PR_AUC per epoch + times + ckpt paths + final arrows."""
    if not log_path or not log_path.exists():
        return {}, {}, {}, {}
    per_epoch_prauc = defaultdict(dict)    # epoch -> split -> prauc
    epoch_times = {}                        # epoch -> minutes
    final_best = {}                         # split -> prauc
    best_ckpts = []                         # list of {"split":..., "path":...}
    cur_epoch = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = EPOCH_HDR.search(line)
        if m:
            cur_epoch = int(m.group(1))  # explicit epoch like "Epoch 7/30"
        m = METRIC_LINE.search(line)
        if m and cur_epoch is not None:
            split = m.group("split").lower()
            prauc = safe_float(m.group("prauc"))
            per_epoch_prauc[cur_epoch][split] = prauc
        m = EPOCH_TIME.search(line)
        if m and cur_epoch is not None:
            epoch_times[cur_epoch] = safe_float(m.group(1))
            cur_epoch = None  # boundary
        m = FINAL_ARROW.search(line)
        if m:
            final_best[m.group("split").lower()] = safe_float(m.group("prauc"))
        m = BEST_SAVE.search(line)
        if m:
            best_ckpts.append({"split": m.group(1).lower(), "path": m.group(2)})
    return per_epoch_prauc, epoch_times, final_best, best_ckpts

def choose_best_epoch(metrics_by_epoch_split, key):
    best_ep, best_val, best_split = None, None, None
    for ep, by_split in metrics_by_epoch_split.items():
        for split, md in by_split.items():
            if key in md and md[key] is not None:
                v = md[key]
                if best_val is None or v > best_val:
                    best_val, best_ep, best_split = v, ep, split
    return best_ep, best_split, best_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder containing metrics_log.csv (i.e., your args.log_dir)")
    ap.add_argument("--log", default=None, help="nohup.out path for PR_AUC per epoch and epoch times")
    ap.add_argument("--base_rate", type=float, default=None, help="Positive rate p (e.g., 0.0529) for lift/normalized-AUPRC")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read metrics_log.csv (AUROC/F1/Precision/Recall/Accuracy/Loss)
    rows = read_metrics_csv(run_dir)

    # Organize by epoch & split
    by_epoch_split = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_epoch_split[r["epoch"]][r["split"]].update({
            "AUROC": r["AUROC"], "F1": r["F1"],
            "Precision": r["Precision"], "Recall": r["Recall"],
            "Accuracy": r["Accuracy"], "Loss": r["Loss"],
        })

    # 2) Parse nohup.out for PR_AUC per epoch + epoch times + final best + ckpts
    log_path = Path(args.log) if args.log else None
    prauc_from_log, epoch_times, final_best_prauc, best_ckpts = parse_nohup(log_path) if log_path else ({}, {}, {}, [])

    # Merge PR_AUC back into by_epoch_split
    for ep, d in prauc_from_log.items():
        for split, pra in d.items():
            by_epoch_split[ep][split]["PR_AUC"] = pra

    # 3) Write merged metrics.csv
    metrics_csv = out_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","split","Loss","AUROC","PR_AUC","F1","Precision","Recall","Accuracy"])
        for ep in sorted(by_epoch_split.keys()):
            for split in SPLITS:
                md = by_epoch_split[ep].get(split, {})
                w.writerow([
                    ep, split,
                    md.get("Loss"), md.get("AUROC"), md.get("PR_AUC"),
                    md.get("F1"), md.get("Precision"), md.get("Recall"), md.get("Accuracy")
                ])

    # 4) Compute bests (prefer PR_AUC if present; else AUROC)
    # Best PR_AUC per split
    best_pr = {s: (None, None) for s in SPLITS}  # split -> (epoch, value)
    for ep, d in by_epoch_split.items():
        for s in SPLITS:
            v = d.get(s, {}).get("PR_AUC")
            if v is not None and (best_pr[s][1] is None or v > best_pr[s][1]):
                best_pr[s] = (ep, v)

    # Best AUROC per split (for reference)
    best_auc = {s: (None, None) for s in SPLITS}
    for ep, d in by_epoch_split.items():
        for s in SPLITS:
            v = d.get(s, {}).get("AUROC")
            if v is not None and (best_auc[s][1] is None or v > best_auc[s][1]):
                best_auc[s] = (ep, v)

    # Fallback: if no per-epoch PR_AUC parsed, use final arrow (“🔹 split → value”)
    for s in ("val","test"):
        if best_pr[s][1] is None and s in final_best_prauc and final_best_prauc[s] is not None:
            # we don't know the epoch, but record the value
            best_pr[s] = (None, final_best_prauc[s])

    # 5) Pull last-epoch operating points/top-k if available
    ops = {}
    for s in ("val","test"):
        p = run_dir / f"ops_{s}.json"
        if p.exists():
            try:
                ops[s] = json.loads(p.read_text())
            except Exception:
                pass

    # 6) Summaries + lifts
    base_rate = args.base_rate
    def lift(v): return (v / base_rate) if (v is not None and base_rate) else None
    def norm_auprc(v): 
        return ((v - base_rate) / (1.0 - base_rate)) if (v is not None and base_rate is not None) else None

    best_test_epoch, best_test_pra = best_pr["test"]
    summary = {
        "run_dir": str(run_dir),
        "log": str(log_path) if log_path else None,
        "epochs_parsed": sorted(by_epoch_split.keys()),
        "epoch_times_minutes": epoch_times,  # may be partial if log omitted
        "total_time_minutes": sum(epoch_times.values()) if epoch_times else None,
        "best_pr_auc": {
            "val": {"epoch": best_pr["val"][0], "value": best_pr["val"][1]},
            "test": {"epoch": best_test_epoch, "value": best_test_pra,
                     "lift_over_random": lift(best_test_pra),
                     "normalized_auprc": norm_auprc(best_test_pra),
                     "random_prauc": base_rate},
        },
        "best_auroc": {
            "val": {"epoch": best_auc["val"][0], "value": best_auc["val"][1]},
            "test": {"epoch": best_auc["test"][0], "value": best_auc["test"][1]},
        },
        "ops_last_epoch": ops,      # last-epoch OP and TopK (if files exist)
        "final_arrow_best": final_best_prauc,  # “🔹 split → …” summary at end of training
        "best_checkpoints": best_ckpts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 7) Markdown report
    md = []
    md.append(f"# Run Report\n\n**Run dir:** `{run_dir}`\n")
    if log_path: md.append(f"**Log:** `{log_path}`\n")
    if summary["total_time_minutes"] is not None:
        md.append(f"**Total epoch time**: ~{summary['total_time_minutes']:.1f} min\n")
    md.append("\n## Best scores\n")
    bpv = summary["best_pr_auc"]["val"]
    bpt = summary["best_pr_auc"]["test"]
    bav = summary["best_auroc"]["val"]
    bat = summary["best_auroc"]["test"]
    md.append(f"- **Val PR-AUC**: {bpv['value']:.4f}" + (f" @ epoch {bpv['epoch']}" if bpv['epoch'] else "") + "\n" if bpv['value'] is not None else "- Val PR-AUC: n/a\n")
    if bpt["value"] is not None:
        md.append(f"- **Test PR-AUC**: {bpt['value']:.4f}" + (f" @ epoch {bpt['epoch']}" if bpt['epoch'] else "") + "\n")
        if base_rate is not None:
            md.append(f"  - Random PR-AUC (p={base_rate:.4f}): {base_rate:.4f}\n")
            if bpt["lift_over_random"] is not None:
                md.append(f"  - Lift over random: {bpt['lift_over_random']:.2f}×\n")
            if bpt["normalized_auprc"] is not None:
                md.append(f"  - Normalized AUPRC: {bpt['normalized_auprc']:.4f}\n")
    else:
        md.append("- Test PR-AUC: n/a\n")
    md.append(f"- **Val AUROC**: {bav['value']:.4f} @ epoch {bav['epoch']}\n" if bav['value'] is not None else "- Val AUROC: n/a\n")
    md.append(f"- **Test AUROC**: {bat['value']:.4f} @ epoch {bat['epoch']}\n" if bat['value'] is not None else "- Test AUROC: n/a\n")

    if summary["best_checkpoints"]:
        md.append("\n## Saved best checkpoints\n")
        for b in summary["best_checkpoints"]:
            md.append(f"- {b['split']}: {b['path']}\n")

    if summary["ops_last_epoch"]:
        md.append("\n## Last-epoch operating points (from ops_*.json)\n")
        for s, d in summary["ops_last_epoch"].items():
            op = d.get("OP", {})
            topk = d.get("TopK", {})
            if op:
                b = op.get("BestF1", {})
                pr20 = op.get("R_at_P>=0.20", {})
                pr60 = op.get("P_at_R>=0.60", {})
                md.append(f"- **{s}**  \n")
                if b: md.append(f"  - BestF1: thr={b.get('thr')}  P={b.get('Precision'):.3f} R={b.get('Recall'):.3f} F1={b.get('F1'):.3f}\n")
                if pr20: md.append(f"  - R@P≥0.20: P={pr20.get('Precision'):.3f} R={pr20.get('Recall'):.3f} F1={pr20.get('F1'):.3f}\n")
                if pr60: md.append(f"  - P@R≥0.60: P={pr60.get('Precision'):.3f} R={pr60.get('Recall'):.3f} F1={pr60.get('F1'):.3f}\n")
            if topk:
                md.append(f"  - Top-k: P@5={topk.get('Prec@5',0):.3f}, R@5={topk.get('Rec@5',0):.3f}, P@10={topk.get('Prec@10',0):.3f}, R@10={topk.get('Rec@10',0):.3f}\n")

    (out_dir / "report.md").write_text("".join(md))

    # 8) Plot curves if possible
    if _HAVE_MPL:
        xs = sorted(by_epoch_split.keys())
        def series(split, key):
            return [by_epoch_split[e].get(split, {}).get(key) for e in xs]

        plt.figure(figsize=(9,6))
        plotted = False
        # PR_AUC if present
        for split in ("val","test"):
            y = series(split, "PR_AUC")
            if any(v is not None for v in y):
                plt.plot(xs, y, label=f"{split} PR_AUC", marker="o"); plotted = True
        # AUROC
        for split in ("val","test"):
            y = series(split, "AUROC")
            if any(v is not None for v in y):
                plt.plot(xs, y, label=f"{split} AUROC", linestyle="--"); plotted = True
        # Loss
        y = series("train","Loss")
        if any(v is not None for v in y):
            plt.plot(xs, y, label="train loss", linestyle=":"); plotted = True
        plt.xlabel("Epoch")
        if plotted: plt.legend()
        plt.title("Training Curves")
        plt.tight_layout()
        plt.savefig(out_dir / "curves.png"); plt.close()

    print(f"[OK] wrote:\n  - {metrics_csv}\n  - {out_dir/'summary.json'}\n  - {out_dir/'report.md'}\n  - {out_dir/'curves.png' if _HAVE_MPL else '(no plot)'}")

if __name__ == "__main__":
    main()
