"""
Analyze per-image watermark results across different datasets.

Reads JSONL log files produced by `run_gaussian_shading.py --per_image_log`
and generates analysis reports to identify content-dependent failure patterns.

Usage:
    python analyze_dataset_results.py --log_dir ./output/
    python analyze_dataset_results.py --log_files ./output/run1/per_image_log.jsonl ./output/run2/per_image_log.jsonl
"""

import argparse
import json
import os
import csv
from collections import defaultdict
from statistics import mean, stdev, median


def load_jsonl_logs(log_paths):
    """Load per-image results from one or more JSONL files."""
    records = []
    for path in log_paths:
        if not os.path.isfile(path):
            print(f"[WARN] File not found: {path}")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def find_log_files(log_dir):
    """Recursively find all per_image_log.jsonl files under a directory."""
    log_files = []
    for root, dirs, files in os.walk(log_dir):
        for fname in files:
            if fname == 'per_image_log.jsonl':
                log_files.append(os.path.join(root, fname))
    return sorted(log_files)


def stats(values):
    """Compute summary statistics for a list of values."""
    if not values:
        return {'count': 0, 'mean': None, 'std': None, 'median': None, 'min': None, 'max': None}
    return {
        'count': len(values),
        'mean': round(mean(values), 6),
        'std': round(stdev(values), 6) if len(values) > 1 else 0.0,
        'median': round(median(values), 6),
        'min': round(min(values), 6),
        'max': round(max(values), 6),
    }


def analyze_by_dataset(records):
    """Group results by dataset and compute per-dataset statistics."""
    by_dataset = defaultdict(list)
    for r in records:
        ds = r.get('dataset', 'unknown')
        # Shorten dataset name for display
        if '/' in ds:
            ds_short = ds.split('/')[-1]
        elif os.sep in ds:
            ds_short = os.path.basename(ds)
        else:
            ds_short = ds
        by_dataset[ds_short].append(r['bit_acc'])
    return {ds: stats(accs) for ds, accs in sorted(by_dataset.items())}


def analyze_by_category(records):
    """Group results by category (if available) and compute statistics."""
    by_cat = defaultdict(list)
    for r in records:
        cat = r.get('category', None)
        if cat:
            by_cat[cat].append(r['bit_acc'])
    if not by_cat:
        return {}
    return {cat: stats(accs) for cat, accs in sorted(by_cat.items())}


def analyze_by_prompt_length(records, buckets=None):
    """Bucket prompts by word count and analyze bit accuracy per bucket."""
    if buckets is None:
        buckets = [(0, 0), (1, 5), (6, 15), (16, 30), (31, 60), (61, 100), (101, 999)]

    by_bucket = defaultdict(list)
    for r in records:
        plen = r.get('prompt_len', 0)
        for lo, hi in buckets:
            if lo <= plen <= hi:
                label = f"{lo}-{hi} words"
                by_bucket[label].append(r['bit_acc'])
                break

    return {bucket: stats(accs) for bucket, accs in sorted(by_bucket.items(),
            key=lambda x: int(x[0].split('-')[0]))}


def find_worst_prompts(records, n=20):
    """Find the prompts with the lowest bit accuracy."""
    sorted_records = sorted(records, key=lambda r: r['bit_acc'])
    return sorted_records[:n]


def print_report(records, output_csv=None):
    """Print a comprehensive analysis report."""
    print(f"\n{'='*80}")
    print(f"  DATASET SENSITIVITY ANALYSIS REPORT")
    print(f"  Total records: {len(records)}")
    print(f"{'='*80}")

    # 1. By Dataset
    ds_stats = analyze_by_dataset(records)
    if ds_stats:
        print(f"\n{'─'*80}")
        print(f"  1. BIT ACCURACY BY DATASET")
        print(f"{'─'*80}")
        print(f"  {'Dataset':<35} {'Count':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*75}")
        for ds, s in ds_stats.items():
            status = ""
            if s['mean'] is not None:
                if s['mean'] < 0.90:
                    status = " *** FAILURE"
                elif s['mean'] < 0.95:
                    status = " ** DEGRADED"
            print(f"  {ds:<35} {s['count']:>6} {s['mean']:>8.4f} {s['std']:>8.4f} "
                  f"{s['min']:>8.4f} {s['max']:>8.4f}{status}")

    # 2. By Category
    cat_stats = analyze_by_category(records)
    if cat_stats:
        print(f"\n{'─'*80}")
        print(f"  2. BIT ACCURACY BY CATEGORY")
        print(f"{'─'*80}")
        print(f"  {'Category':<30} {'Count':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*70}")
        for cat, s in sorted(cat_stats.items(), key=lambda x: x[1]['mean'] or 1.0):
            status = ""
            if s['mean'] is not None:
                if s['mean'] < 0.90:
                    status = " *** FAILURE"
                elif s['mean'] < 0.95:
                    status = " ** DEGRADED"
            print(f"  {cat:<30} {s['count']:>6} {s['mean']:>8.4f} {s['std']:>8.4f} "
                  f"{s['min']:>8.4f} {s['max']:>8.4f}{status}")

    # 3. By Prompt Length
    len_stats = analyze_by_prompt_length(records)
    if len_stats:
        print(f"\n{'─'*80}")
        print(f"  3. BIT ACCURACY BY PROMPT LENGTH")
        print(f"{'─'*80}")
        print(f"  {'Length Bucket':<20} {'Count':>6} {'Mean':>8} {'Std':>8} {'Min':>8}")
        print(f"  {'-'*50}")
        for bucket, s in len_stats.items():
            if s['mean'] is not None:
                print(f"  {bucket:<20} {s['count']:>6} {s['mean']:>8.4f} {s['std']:>8.4f} {s['min']:>8.4f}")

    # 4. Worst Prompts
    worst = find_worst_prompts(records, n=20)
    if worst:
        print(f"\n{'─'*80}")
        print(f"  4. TOP 20 WORST PROMPTS (lowest bit accuracy)")
        print(f"{'─'*80}")
        for i, r in enumerate(worst):
            prompt_short = r['prompt'][:80] + '...' if len(r['prompt']) > 80 else r['prompt']
            cat = r.get('category', '-')
            ds = r.get('dataset', '-')
            if '/' in ds:
                ds = ds.split('/')[-1]
            print(f"  {i+1:>3}. [{r['bit_acc']:.4f}] (dataset={ds}, cat={cat})")
            print(f"       \"{prompt_short}\"")

    # 5. Export CSV
    if output_csv:
        rows = []
        for ds, s in ds_stats.items():
            rows.append({'group_type': 'dataset', 'group_name': ds, **s})
        for cat, s in cat_stats.items():
            rows.append({'group_type': 'category', 'group_name': cat, **s})
        for bucket, s in len_stats.items():
            rows.append({'group_type': 'prompt_length', 'group_name': bucket, **s})

        fieldnames = ['group_type', 'group_name', 'count', 'mean', 'std', 'median', 'min', 'max']
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[INFO] CSV summary saved to: {output_csv}")

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze per-image watermark results across datasets',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--log_dir', default=None,
                        help='Directory to recursively search for per_image_log.jsonl files')
    parser.add_argument('--log_files', nargs='+', default=None,
                        help='Specific JSONL log files to analyze')
    parser.add_argument('--output_csv', default=None,
                        help='Optional: save summary statistics to CSV file')

    args = parser.parse_args()

    # Collect log files
    log_files = []
    if args.log_files:
        log_files = args.log_files
    elif args.log_dir:
        log_files = find_log_files(args.log_dir)
    else:
        # Default: search current directory
        log_files = find_log_files('./output/')
        if not log_files:
            log_files = find_log_files('./')

    if not log_files:
        print("[ERROR] No per_image_log.jsonl files found.")
        print("  Run experiments with --per_image_log first, or specify --log_dir / --log_files.")
        return

    print(f"[INFO] Found {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  - {f}")

    records = load_jsonl_logs(log_files)
    if not records:
        print("[ERROR] No valid records found in log files.")
        return

    print_report(records, output_csv=args.output_csv)


if __name__ == '__main__':
    main()
