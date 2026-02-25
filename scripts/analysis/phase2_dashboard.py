"""
Phase 2 Self-Supervised Training Dashboard.

Monitors z-space quality, holdout trends, mode collapse detection, and
Phase 2 loss convergence. Reads from outputs/phase2_log.csv and
outputs/training_log.csv.

Usage:
    python scripts/analysis/phase2_dashboard.py [--output outputs/phase2_dashboard.png]
    python scripts/analysis/phase2_dashboard.py --text-only  # No matplotlib needed

Reference: docs/PHASE2_SELF_SUPERVISED_DESIGN.md

February 2026
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_phase2_log(log_path: Path) -> list:
    """Load Phase 2 metrics from CSV."""
    if not log_path.exists():
        return []
    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if '.' in str(v) else int(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_training_log(log_path: Path) -> list:
    """Load main training metrics from CSV."""
    if not log_path.exists():
        return []
    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if '.' in str(v) else int(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_holdout_results(output_dir: Path) -> list:
    """Load holdout mini search results (multiple JSONs)."""
    results = []
    for json_path in sorted(output_dir.glob('holdout_mini_*.json')):
        with open(json_path) as f:
            data = json.load(f)
            results.append(data)
    return results


def print_text_dashboard(phase2_rows, training_rows, holdout_results):
    """Print a text-based dashboard to stdout."""
    print("=" * 70)
    print("PHASE 2 SELF-SUPERVISED TRAINING DASHBOARD")
    print("=" * 70)

    if not phase2_rows:
        print("\n  No Phase 2 data found (outputs/phase2_log.csv)")
        print("  Phase 2 may not have been enabled or activated yet.")
        return

    latest = phase2_rows[-1]
    first = phase2_rows[0]

    print(f"\n{'OVERVIEW':=^50}")
    print(f"  Phase 2 sub-epochs: {len(phase2_rows)}")
    print(f"  Epoch range: {first.get('epoch', '?')} → {latest.get('epoch', '?')}")
    print(f"  Current weight: {latest.get('phase2_weight', 0):.4f}")

    # Loss trends
    print(f"\n{'LOSS SIGNALS':=^50}")
    print(f"  {'Metric':<25} {'First':>10} {'Latest':>10} {'Trend':>10}")
    print(f"  {'-'*55}")

    loss_keys = [
        ('phase2_total_loss', 'Total Loss'),
        ('phase2_loss1_rt', 'L1: Round-Trip'),
        ('phase2_loss2_consist', 'L2: Consistency'),
        ('phase2_loss3_physics', 'L3: Physics'),
        ('phase2_loss4_reinforce', 'L4: REINFORCE'),
        ('phase2_z_mse', 'Z MSE'),
        ('phase2_tc_mse', 'Tc MSE'),
    ]

    for key, label in loss_keys:
        first_val = first.get(key, 0)
        latest_val = latest.get(key, 0)
        if first_val > 0:
            trend = (latest_val - first_val) / first_val * 100
            trend_str = f"{trend:+.1f}%"
        else:
            trend_str = "N/A"
        print(f"  {label:<25} {first_val:>10.4f} {latest_val:>10.4f} {trend_str:>10}")

    # Quality metrics
    print(f"\n{'Z-SPACE QUALITY':=^50}")
    print(f"  {'Metric':<25} {'First':>10} {'Latest':>10} {'Target':>10}")
    print(f"  {'-'*55}")

    quality_keys = [
        ('phase2_valid_rate', 'Valid Formula Rate', '> 0.80'),
        ('phase2_unique_rate', 'Unique Formula Rate', '> 0.30'),
        ('phase2_n_valid', 'Valid Count', '-'),
        ('phase2_n_unique_formulas', 'Unique Formulas', '-'),
        ('phase2_n_degenerate', 'Degenerate (diag.)', '-'),
    ]

    for key, label, target in quality_keys:
        first_val = first.get(key, 0)
        latest_val = latest.get(key, 0)
        print(f"  {label:<25} {first_val:>10.3f} {latest_val:>10.3f} {target:>10}")

    # Filter pipeline
    print(f"\n{'FILTER PIPELINE (Latest)':=^50}")
    filter_keys = [
        'phase2_filter_total', 'phase2_filter_pass_parse',
        'phase2_filter_pass_candidate', 'phase2_filter_pass_physics',
        'phase2_filter_pass_constraints',
    ]
    labels = ['Total Generated', 'Pass Parse', 'Pass Candidate', 'Pass Physics', 'Pass Constraints']
    for key, label in zip(filter_keys, labels):
        val = latest.get(key, 0)
        print(f"  {label:<25} {val:>6}")

    # Sampling strategy
    print(f"\n{'SAMPLING STRATEGY (Latest)':=^50}")
    for key, label in [('phase2_sample_n_perturb', 'Perturbation'),
                       ('phase2_sample_n_slerp', 'SLERP'),
                       ('phase2_sample_n_pca', 'PCA Walk')]:
        val = latest.get(key, 0)
        print(f"  {label:<25} {val:>6}")

    # Mode collapse
    collapse_active = latest.get('phase2_collapse_active', 0)
    print(f"\n{'SAFETY STATUS':=^50}")
    print(f"  Mode collapse: {'ACTIVE' if collapse_active else 'OK'}")
    print(f"  Current weight: {latest.get('phase2_weight', 0):.4f}")

    # Novel discoveries (cumulative across all sub-epochs)
    total_novel = sum(r.get('phase2_n_novel', 0) for r in phase2_rows)
    total_holdout_rec = sum(r.get('phase2_n_holdout_recovered', 0) for r in phase2_rows)
    if total_novel > 0 or total_holdout_rec > 0:
        print(f"\n{'DISCOVERIES (opportunistic)':=^50}")
        print(f"  Novel formulas flagged:  {total_novel}")
        print(f"  Holdout recoveries:      {total_holdout_rec}")
        print(f"  See: outputs/phase2_discoveries.jsonl")

    # Training exact match trend (from main log)
    if training_rows:
        recent = training_rows[-10:] if len(training_rows) >= 10 else training_rows
        exact_values = [r.get('exact_match', 0) for r in recent]
        if exact_values:
            print(f"  Training exact (last 10): {min(exact_values):.3f} → {max(exact_values):.3f}")

    # Holdout results
    if holdout_results:
        print(f"\n{'HOLDOUT DISCOVERY':=^50}")
        for hr in holdout_results[-3:]:  # Last 3 holdout searches
            n_exact = hr.get('n_exact', 0)
            n_095 = hr.get('n_found_095', 0)
            n_targets = hr.get('n_targets_searched', 45)
            ts = hr.get('timestamp', '?')
            print(f"  [{ts}] Exact: {n_exact}/{n_targets}, "
                  f">=0.95: {n_095}/{n_targets}")

    # Milestones
    print(f"\n{'MILESTONES':=^50}")
    z_mse = latest.get('phase2_z_mse', float('inf'))
    milestones = [
        ('Alpha', 'Round-trip Z-MSE < 0.1', z_mse < 0.1),
    ]
    for name, desc, achieved in milestones:
        status = 'ACHIEVED' if achieved else 'pending'
        print(f"  Phase 2 {name}: {desc} → [{status}]")

    print(f"\n{'='*70}")


def plot_dashboard(phase2_rows, training_rows, holdout_results, output_path):
    """Generate matplotlib dashboard plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Use --text-only for text dashboard.")
        return

    if not phase2_rows:
        print("No Phase 2 data to plot.")
        return

    epochs = [r.get('epoch', 0) for r in phase2_rows]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Phase 2 Self-Supervised Training Dashboard', fontsize=14, fontweight='bold')

    # Panel 1: Loss signals
    ax = axes[0, 0]
    for key, label, color in [
        ('phase2_loss1_rt', 'L1: Round-Trip', 'blue'),
        ('phase2_loss2_consist', 'L2: Consistency', 'orange'),
        ('phase2_loss3_physics', 'L3: Physics', 'green'),
        ('phase2_loss4_reinforce', 'L4: REINFORCE', 'red'),
    ]:
        values = [r.get(key, 0) for r in phase2_rows]
        ax.plot(epochs, values, label=label, color=color, alpha=0.8)
    ax.set_ylabel('Loss')
    ax.set_title('Phase 2 Loss Signals')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Z-MSE and Tc-MSE
    ax = axes[0, 1]
    z_mse = [r.get('phase2_z_mse', 0) for r in phase2_rows]
    tc_mse = [r.get('phase2_tc_mse', 0) for r in phase2_rows]
    ax.plot(epochs, z_mse, label='Z MSE', color='blue')
    ax.plot(epochs, tc_mse, label='Tc MSE', color='red')
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Alpha target')
    ax.set_ylabel('MSE')
    ax.set_title('Round-Trip Reconstruction Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Valid and unique rates
    ax = axes[1, 0]
    valid_rate = [r.get('phase2_valid_rate', 0) for r in phase2_rows]
    unique_rate = [r.get('phase2_unique_rate', 0) for r in phase2_rows]
    ax.plot(epochs, valid_rate, label='Valid Rate', color='blue')
    ax.plot(epochs, unique_rate, label='Unique Rate', color='orange')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')
    ax.set_ylabel('Rate')
    ax.set_title('Generation Quality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 4: Phase 2 weight and total loss
    ax = axes[1, 1]
    weight = [r.get('phase2_weight', 0) for r in phase2_rows]
    total = [r.get('phase2_total_loss', 0) for r in phase2_rows]
    ax2 = ax.twinx()
    ax.plot(epochs, weight, label='Weight', color='blue')
    ax2.plot(epochs, total, label='Total Loss', color='red', alpha=0.7)
    ax.set_ylabel('Weight', color='blue')
    ax2.set_ylabel('Total Loss', color='red')
    ax.set_title('Phase 2 Weight & Total Loss')
    ax.grid(True, alpha=0.3)

    # Panel 5: Filter pipeline
    ax = axes[2, 0]
    for key, label, color in [
        ('phase2_filter_pass_parse', 'Parse', 'blue'),
        ('phase2_filter_pass_candidate', 'Candidate', 'orange'),
        ('phase2_filter_pass_physics', 'Physics', 'green'),
        ('phase2_filter_pass_constraints', 'Constraints', 'red'),
    ]:
        values = [r.get(key, 0) for r in phase2_rows]
        ax.plot(epochs, values, label=label, color=color, alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_xlabel('Epoch')
    ax.set_title('Filter Pipeline Pass Counts')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: Training exact match (from main log)
    ax = axes[2, 1]
    if training_rows:
        train_epochs = [r.get('epoch', 0) for r in training_rows]
        exact_match = [r.get('exact_match', 0) for r in training_rows]
        ax.plot(train_epochs, exact_match, color='blue', alpha=0.5, linewidth=0.5)
        # Smoothed
        window = min(20, len(exact_match) // 5) if len(exact_match) > 20 else 1
        if window > 1:
            smoothed = []
            for i in range(len(exact_match)):
                start = max(0, i - window)
                smoothed.append(sum(exact_match[start:i+1]) / len(exact_match[start:i+1]))
            ax.plot(train_epochs, smoothed, color='blue', linewidth=1.5, label='Smoothed')
    ax.set_ylabel('Exact Match')
    ax.set_xlabel('Epoch')
    ax.set_title('Training Exact Match')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Dashboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Self-Supervised Training Dashboard')
    parser.add_argument('--output', type=str, default='outputs/phase2_dashboard.png',
                        help='Output path for dashboard plot')
    parser.add_argument('--text-only', action='store_true',
                        help='Print text dashboard only (no matplotlib needed)')
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / 'outputs'
    phase2_log_path = output_dir / 'phase2_log.csv'
    training_log_path = output_dir / 'training_log.csv'

    phase2_rows = load_phase2_log(phase2_log_path)
    training_rows = load_training_log(training_log_path)
    holdout_results = load_holdout_results(output_dir)

    # Always print text dashboard
    print_text_dashboard(phase2_rows, training_rows, holdout_results)

    # Generate plot unless text-only
    if not args.text_only and phase2_rows:
        output_path = PROJECT_ROOT / args.output
        plot_dashboard(phase2_rows, training_rows, holdout_results, output_path)


if __name__ == '__main__':
    main()
