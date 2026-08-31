"""
Factor management script -- statistics, classification, sorting, filtering, deleting valid factors
Usage:
    python manager.py                        # Default statistics overview
    python manager.py --sort best_auc        # Sort by highest AUC
    python manager.py --sort weighted_auc    # Sort by weighted AUC of valid classes
    python manager.py --seq 1 5              # Only show seq_length in [1,5]
    python manager.py --class climbsocial    # Only show factors valid for a specific class
    python manager.py --top 20               # Only show top N
    python manager.py --delete NAME          # Delete specified factor (requires confirmation)
    python manager.py --export out.json      # Export filtered results
    python manager.py --stats                # Detailed statistics table
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import argparse
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

# Windows terminal Chinese display fix
if sys.platform == "win32":
    import io
    if getattr(sys.stdout, "encoding", None) != "utf-8" and getattr(sys.stdout, "buffer", None) is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FACTORS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory_before", "valid_factors.json")

LABEL_MAP = {
    "0": "explore_object",
    "1": "climb",
    "2": "self_grooming",
    "3": "stand",
    "4": "blank",
    "5": "positive_sniffs",
    "6": "approach",
    "7": "climbsocial",
}


# --------------------------- helpers ----------------------------

def load_factors(path=FACTORS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_factors(factors, path=FACTORS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(factors, f, ensure_ascii=False, indent=2)


def class_label(cls_id: str) -> str:
    return LABEL_MAP.get(str(cls_id), str(cls_id))


def weighted_auc(factor: dict) -> float:
    """Simple mean of valid class AUCs (equal weight)."""
    vcs = factor.get("valid_classes", [])
    if not vcs:
        return factor.get("best_auc", 0.0)
    return sum(vc["auc"] for vc in vcs) / len(vcs)


def sort_key(factor: dict, method: str):
    if method == "best_auc":
        return factor.get("best_auc", 0.0)
    if method == "weighted_auc":
        return weighted_auc(factor)
    if method == "n_valid":
        return len(factor.get("valid_classes", []))
    if method == "seq_length":
        return factor.get("seq_length", 0)
    if method == "saved_at":
        return factor.get("saved_at", "")
    return factor.get("best_auc", 0.0)


def filter_factors(factors, seq_lengths=None, target_class=None, min_auc=None):
    result = factors
    if seq_lengths:
        result = [f for f in result if f.get("seq_length") in seq_lengths]
    if target_class is not None:
        # Support class name or numeric ID
        cls_id = None
        for k, v in LABEL_MAP.items():
            if v == target_class or k == str(target_class):
                cls_id = k
                break
        if cls_id is None:
            cls_id = str(target_class)
        result = [
            f for f in result
            if any(str(vc["class"]) == cls_id for vc in f.get("valid_classes", []))
        ]
    if min_auc is not None:
        result = [f for f in result if f.get("best_auc", 0) >= min_auc]
    return result


# --------------------------- display ----------------------------

def fmt_valid_classes(vcs):
    parts = []
    for vc in vcs:
        name = class_label(vc["class"])
        auc = vc["auc"]
        f1 = vc.get("f1")
        if f1 is not None:
            parts.append(f"{name}(AUC={auc:.4f},F1={f1:.4f})")
        else:
            parts.append(f"{name}(AUC={auc:.4f})")
    return ", ".join(parts)


def print_factor_table(factors, sort_by="best_auc", top=None):
    sorted_f = sorted(factors, key=lambda f: sort_key(f, sort_by), reverse=True)
    if top:
        sorted_f = sorted_f[:top]

    col_w = [4, 48, 6, 8, 8, 8, 60]
    header = ["#", "Name", "Seq", "BestAUC", "WgtAUC", "NValid", "ValidClasses"]
    sep = "  ".join("-" * w for w in col_w)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print(fmt.format(*header))
    print(sep)
    for i, f in enumerate(sorted_f, 1):
        wauc = weighted_auc(f)
        vcs_str = fmt_valid_classes(f.get("valid_classes", []))
        name = f.get("name", "")
        if len(name) > col_w[1]:
            name = name[:col_w[1] - 1] + "..."
        print(fmt.format(
            str(i),
            name,
            str(f.get("seq_length", "?")),
            f"{f.get('best_auc', 0):.4f}",
            f"{wauc:.4f}",
            str(len(f.get("valid_classes", []))),
            vcs_str[:col_w[6]],
        ))
    print(f"\nTotal {len(sorted_f)} factors (sorted: {sort_by})")


def print_stats(factors):
    total = len(factors)
    print(f"{'='*60}")
    print(f"  Total factors: {total}")
    print(f"{'='*60}")

    # Group by seq_length
    by_seq = defaultdict(list)
    for f in factors:
        by_seq[f.get("seq_length", "?")].append(f)

    print("\n-- Grouped by seq_length --")
    print(f"  {'seq_len':<10} {'Count':<8} {'AvgBestAUC':<14} {'AvgWgtAUC':<14}")
    for seq in sorted(by_seq.keys(), key=lambda x: (x is None, x)):
        grp = by_seq[seq]
        avg_best = sum(f.get("best_auc", 0) for f in grp) / len(grp)
        avg_wgt = sum(weighted_auc(f) for f in grp) / len(grp)
        print(f"  {str(seq):<10} {len(grp):<8} {avg_best:<14.4f} {avg_wgt:<14.4f}")

    # By valid class statistics
    print("\n-- By valid class (how many factors are valid for each class) --")
    class_count = defaultdict(int)
    class_auc_sum = defaultdict(float)
    for f in factors:
        for vc in f.get("valid_classes", []):
            cid = str(vc["class"])
            class_count[cid] += 1
            class_auc_sum[cid] += vc["auc"]

    print(f"  {'Class':<20} {'ValidFactors':<12} {'AvgAUC':<10}")
    for cid in sorted(class_count.keys(), key=lambda x: -class_count[x]):
        cnt = class_count[cid]
        avg = class_auc_sum[cid] / cnt
        print(f"  {class_label(cid):<20} {cnt:<12} {avg:.4f}")

    # Cross-tabulation: seq_length x valid class
    print("\n-- seq_length x valid class cross count --")
    seq_list = sorted(by_seq.keys(), key=lambda x: (x is None, x))
    cls_list = sorted(class_count.keys())
    header = f"  {'seq':<8}" + "".join(f"{class_label(c)[:12]:<14}" for c in cls_list)
    print(header)
    for seq in seq_list:
        grp = by_seq[seq]
        row = f"  {str(seq):<8}"
        for cid in cls_list:
            cnt = sum(
                1 for f in grp
                if any(str(vc["class"]) == cid for vc in f.get("valid_classes", []))
            )
            row += f"{cnt:<14}"
        print(row)

    # Mode distribution
    print("\n-- Factor mode distribution --")
    mode_count = defaultdict(int)
    for f in factors:
        mode_count[f.get("mode", "row")] += 1
    for mode, cnt in sorted(mode_count.items()):
        print(f"  {mode}: {cnt}")

    # AUC distribution bins
    print("\n-- BestAUC distribution --")
    bins = [(0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]
    for lo, hi in bins:
        cnt = sum(1 for f in factors if lo <= f.get("best_auc", 0) < hi)
        bar = "#" * cnt
        print(f"  [{lo:.2f},{hi:.2f}): {cnt:>4}  {bar}")

    print(f"\n  Max BestAUC: {max(f.get('best_auc',0) for f in factors):.4f}")
    print(f"  Avg BestAUC: {sum(f.get('best_auc',0) for f in factors)/total:.4f}")
    print(f"  Avg valid classes: {sum(len(f.get('valid_classes',[])) for f in factors)/total:.2f}")


# --------------------------- management -------------------------

def delete_factor(name: str, path=FACTORS_PATH, dry_run=False):
    factors = load_factors(path)
    before = len(factors)
    new_factors = [f for f in factors if f.get("name") != name]
    if len(new_factors) == before:
        print(f"[!] Factor not found: {name}")
        return False
    removed = [f for f in factors if f.get("name") == name]
    print(f"Will delete: {removed[0]['name']}  BestAUC={removed[0].get('best_auc')}")
    if dry_run:
        print("[dry-run] Not written")
        return True
    ans = input("Confirm deletion? (y/N) ").strip().lower()
    if ans != "y":
        print("Cancelled")
        return False
    save_factors(new_factors, path)
    print(f"Deleted, {len(new_factors)} factors remaining")
    return True


def deduplicate(path=FACTORS_PATH, dry_run=False):
    """Deduplicate by factor name, keeping the version with highest best_auc."""
    factors = load_factors(path)
    best = {}
    for f in factors:
        name = f.get("name", "")
        if name not in best or f.get("best_auc", 0) > best[name].get("best_auc", 0):
            best[name] = f
    new_factors = list(best.values())
    removed = len(factors) - len(new_factors)
    print(f"Dedup: original {len(factors)} -> after dedup {len(new_factors)}, removed {removed} duplicates")
    if removed == 0:
        return
    if dry_run:
        print("[dry-run] Not written")
        return
    ans = input("Confirm write? (y/N) ").strip().lower()
    if ans == "y":
        save_factors(new_factors, path)
        print("Written")


# --------------------------- CLI --------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Factor management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--path", default=FACTORS_PATH, help="valid_factors.json path")
    p.add_argument(
        "--sort",
        default="best_auc",
        choices=["best_auc", "weighted_auc", "n_valid", "seq_length", "saved_at"],
        help="Sort method",
    )
    p.add_argument("--seq", nargs="+", type=int, metavar="N", help="Filter by seq_length (multiple allowed)")
    p.add_argument("--class", dest="cls", metavar="CLASS", help="Filter by valid class (name or ID)")
    p.add_argument("--min-auc", type=float, metavar="AUC", help="Minimum best_auc threshold")
    p.add_argument("--top", type=int, metavar="N", help="Only show top N")
    p.add_argument("--stats", action="store_true", help="Output detailed statistics")
    p.add_argument("--delete", metavar="NAME", help="Delete factor with specified name")
    p.add_argument("--dedup", action="store_true", help="Deduplicate by name (keep highest AUC)")
    p.add_argument("--export", metavar="FILE", help="Export filtered results to JSON file")
    p.add_argument("--list-classes", action="store_true", help="List all class IDs and names")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_classes:
        print("Class ID -> Name:")
        for k, v in sorted(LABEL_MAP.items()):
            print(f"  {k}: {v}")
        return

    factors = load_factors(args.path)

    # Management operations
    if args.delete:
        delete_factor(args.delete, args.path)
        return

    if args.dedup:
        deduplicate(args.path)
        return

    # Filtering
    filtered = filter_factors(
        factors,
        seq_lengths=args.seq,
        target_class=args.cls,
        min_auc=args.min_auc,
    )

    if args.stats:
        print_stats(filtered)
        print()

    print_factor_table(filtered, sort_by=args.sort, top=args.top)

    if args.export:
        sorted_f = sorted(filtered, key=lambda f: sort_key(f, args.sort), reverse=True)
        if args.top:
            sorted_f = sorted_f[:args.top]
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(sorted_f, f, ensure_ascii=False, indent=2)
        print(f"\nExported {len(sorted_f)} factors to {args.export}")


if __name__ == "__main__":
    main()
