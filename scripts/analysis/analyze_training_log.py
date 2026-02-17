#!/usr/bin/env python3
"""
Analyze training_log.csv for the Superconductor VAE project.
Tracks key metrics, phase transitions, and trends.

Handles the mid-training column addition:
- Original (18 cols): epoch..hp_loss, true_exact, epoch_time, timestamp
- Transitional (19 cols): added sc_loss after hp_loss
- New (20 cols): added sc_loss + theory_loss after hp_loss
"""

import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = "/mnt/c/Users/james/Downloads/training_log.csv"

# Column definitions for each format
COLS_18 = [
    'epoch', 'exact_match', 'accuracy', 'loss', 'tc_loss', 'magpie_loss',
    'stoich_loss', 'rl_loss', 'reward', 'entropy', 'entropy_weight',
    'z_norm', 'tf_ratio', 'contrastive_loss', 'hp_loss',
    'true_exact', 'epoch_time', 'timestamp'
]

COLS_19 = [
    'epoch', 'exact_match', 'accuracy', 'loss', 'tc_loss', 'magpie_loss',
    'stoich_loss', 'rl_loss', 'reward', 'entropy', 'entropy_weight',
    'z_norm', 'tf_ratio', 'contrastive_loss', 'hp_loss', 'sc_loss',
    'true_exact', 'epoch_time', 'timestamp'
]

COLS_20 = [
    'epoch', 'exact_match', 'accuracy', 'loss', 'tc_loss', 'magpie_loss',
    'stoich_loss', 'rl_loss', 'reward', 'entropy', 'entropy_weight',
    'z_norm', 'tf_ratio', 'contrastive_loss', 'hp_loss', 'sc_loss',
    'theory_loss', 'true_exact', 'epoch_time', 'timestamp'
]

# Unified column set (superset)
ALL_COLS = [
    'epoch', 'exact_match', 'accuracy', 'loss', 'tc_loss', 'magpie_loss',
    'stoich_loss', 'rl_loss', 'reward', 'entropy', 'entropy_weight',
    'z_norm', 'tf_ratio', 'contrastive_loss', 'hp_loss', 'sc_loss',
    'theory_loss', 'true_exact', 'epoch_time', 'timestamp'
]


def load_csv_with_format_change(csv_path):
    """Load CSV that changes column count mid-file."""
    rows = []
    with open(csv_path) as f:
        header_line = f.readline()  # skip header
        for line in f:
            fields = line.strip().split(',')
            n = len(fields)
            if n == 18:
                cols = COLS_18
            elif n == 19:
                cols = COLS_19
            elif n == 20:
                cols = COLS_20
            else:
                # Try best-effort: skip malformed
                continue

            row_dict = {}
            for col_name, val in zip(cols, fields):
                row_dict[col_name] = val
            rows.append(row_dict)

    df = pd.DataFrame(rows, columns=ALL_COLS)

    # Convert numeric columns
    numeric_cols = [c for c in ALL_COLS if c != 'timestamp']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def main():
    # == 1. Read CSV ==
    df = load_csv_with_format_change(CSV_PATH)
    print(f"Loaded {len(df)} rows, epochs {int(df['epoch'].min())} - {int(df['epoch'].max())}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows with sc_loss data: {df['sc_loss'].notna().sum()}")
    print(f"Rows with theory_loss data: {df['theory_loss'].notna().sum()}")
    print()

    # == 2. Peak exact_match ==
    idx_peak_em = df['exact_match'].idxmax()
    peak_em_row = df.loc[idx_peak_em]
    print("=" * 90)
    print("PEAK EXACT_MATCH")
    print("=" * 90)
    print(f"  Epoch:       {int(peak_em_row['epoch'])}")
    print(f"  exact_match: {peak_em_row['exact_match']:.6f}  ({peak_em_row['exact_match']*100:.2f}%)")
    print(f"  true_exact:  {peak_em_row['true_exact']:.6f}  ({peak_em_row['true_exact']*100:.2f}%)")
    print(f"  loss:        {peak_em_row['loss']:.6f}")
    print()

    # == 3. Peak true_exact ==
    idx_peak_te = df['true_exact'].idxmax()
    peak_te_row = df.loc[idx_peak_te]
    print("=" * 90)
    print("PEAK TRUE_EXACT")
    print("=" * 90)
    print(f"  Epoch:       {int(peak_te_row['epoch'])}")
    print(f"  true_exact:  {peak_te_row['true_exact']:.6f}  ({peak_te_row['true_exact']*100:.2f}%)")
    print(f"  exact_match: {peak_te_row['exact_match']:.6f}  ({peak_te_row['exact_match']*100:.2f}%)")
    print(f"  loss:        {peak_te_row['loss']:.6f}")
    print()

    # == 4. Detect when losses went to 0 / near-0 ==
    print("=" * 90)
    print("LOSS TRANSITION DETECTION (when losses dropped to ~0)")
    print("=" * 90)

    loss_cols_to_track = ['contrastive_loss', 'rl_loss', 'hp_loss']
    threshold = 0.01

    for col in loss_cols_to_track:
        if col not in df.columns:
            print(f"  {col}: column not found")
            continue
        vals = df[col].dropna().values
        ep_vals = df.loc[df[col].notna(), 'epoch'].values

        if len(vals) == 0:
            print(f"  {col}: no data")
            continue

        max_val = np.nanmax(vals)
        if max_val < threshold:
            print(f"  {col}: always near-zero (max={max_val:.6f})")
            continue

        above_mask = vals > threshold
        if not above_mask.any():
            print(f"  {col}: always below threshold {threshold}")
            continue

        last_above_idx = np.where(above_mask)[0][-1]
        if last_above_idx < len(vals) - 1:
            trans_ep = int(ep_vals[last_above_idx + 1])
            val_before = vals[last_above_idx]
            val_after = vals[last_above_idx + 1]
            print(f"  {col}:")
            print(f"    Last non-trivial epoch: {int(ep_vals[last_above_idx])} (value={val_before:.6f})")
            print(f"    Dropped to near-0 at epoch: {trans_ep} (value={val_after:.6f})")
            print(f"    Range when active: [{np.nanmin(vals[above_mask]):.6f}, {max_val:.6f}]")
        else:
            print(f"  {col}: still active at latest epoch {int(ep_vals[-1])} (value={vals[-1]:.6f})")

        # Check for intermediate transitions
        transitions = np.diff(above_mask.astype(int))
        drop_indices = np.where(transitions == -1)[0]
        rise_indices = np.where(transitions == 1)[0]
        if len(drop_indices) > 1 or len(rise_indices) > 1:
            print(f"    NOTE: Multiple transitions detected ({len(drop_indices)} drops, {len(rise_indices)} rises)")
            for di in drop_indices[:8]:
                print(f"      Dropped at epoch {int(ep_vals[di+1])}")
            for ri in rise_indices[:8]:
                print(f"      Rose at epoch {int(ep_vals[ri+1])}")

    print()

    # == 5. Phase Analysis ==
    print("=" * 90)
    print("PHASE ANALYSIS: Before vs After Loss Transitions")
    print("=" * 90)

    # Detect transition: when contrastive_loss dropped to ~0
    cl_valid = df.dropna(subset=['contrastive_loss'])
    cl_vals = cl_valid['contrastive_loss'].values
    cl_epochs = cl_valid['epoch'].values
    cl_above = cl_vals > 0.01

    transition_epoch = None
    if cl_above.any():
        last_cl_above_idx = np.where(cl_above)[0][-1]
        if last_cl_above_idx < len(cl_vals) - 1:
            transition_epoch = int(cl_epochs[last_cl_above_idx + 1])

    # If contrastive was always near-zero, look for hp_loss transition
    if transition_epoch is None:
        hp_valid = df.dropna(subset=['hp_loss'])
        hp_vals = hp_valid['hp_loss'].values
        hp_epochs = hp_valid['epoch'].values
        hp_above = hp_vals > 0.01
        if hp_above.any():
            last_hp_above_idx = np.where(hp_above)[0][-1]
            if last_hp_above_idx < len(hp_vals) - 1:
                transition_epoch = int(hp_epochs[last_hp_above_idx + 1])

    metrics_for_phase = ['exact_match', 'true_exact', 'loss', 'tc_loss', 'magpie_loss',
                         'contrastive_loss', 'hp_loss', 'rl_loss', 'z_norm']

    if transition_epoch is not None:
        print(f"\nTransition epoch (contrastive_loss dropped to ~0): ~{transition_epoch}")
        before = df[df['epoch'] < transition_epoch]
        after = df[df['epoch'] >= transition_epoch]

        print(f"  Epochs before transition: {len(before)} rows ({int(before['epoch'].min())}-{int(before['epoch'].max())})")
        print(f"  Epochs after transition:  {len(after)} rows ({int(after['epoch'].min())}-{int(after['epoch'].max())})")

        print(f"\n{'Metric':<20} {'Before (mean)':>15} {'Before (last10)':>16} {'After (mean)':>15} {'After (last10)':>16} {'Delta(last10)':>14}")
        print("-" * 98)
        for m in metrics_for_phase:
            if m not in df.columns:
                continue
            b_mean = before[m].mean()
            b_last10 = before[m].tail(10).mean()
            a_mean = after[m].mean()
            a_last10 = after[m].tail(10).mean()
            delta = a_last10 - b_last10
            print(f"{m:<20} {b_mean:>15.6f} {b_last10:>16.6f} {a_mean:>15.6f} {a_last10:>16.6f} {delta:>+14.6f}")
    else:
        print("  No clear transition point detected - losses may have been constant.")
        print(f"\n{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Last10 avg':>14}")
        print("-" * 82)
        for m in metrics_for_phase:
            if m not in df.columns:
                continue
            print(f"{m:<20} {df[m].mean():>12.6f} {df[m].std():>12.6f} "
                  f"{df[m].min():>12.6f} {df[m].max():>12.6f} {df[m].tail(10).mean():>14.6f}")

    print()

    # == 6. Summary Table at Key Milestones ==
    print("=" * 90)
    print("SUMMARY TABLE AT KEY MILESTONES")
    print("=" * 90)

    milestones = {}
    milestones['Start (first 5)'] = df.head(5).mean(numeric_only=True)
    milestones[f'Peak exact_match (ep {int(peak_em_row["epoch"])})'] = peak_em_row
    milestones[f'Peak true_exact (ep {int(peak_te_row["epoch"])})'] = peak_te_row

    if transition_epoch is not None:
        trans_rows = df[df['epoch'] == transition_epoch]
        if len(trans_rows) > 0:
            milestones[f'Transition (ep {transition_epoch})'] = trans_rows.iloc[0]
        else:
            closest_idx = (df['epoch'] - transition_epoch).abs().idxmin()
            milestones[f'Transition (~ep {int(df.loc[closest_idx, "epoch"])})'] = df.loc[closest_idx]

    milestones['Latest (last 5)'] = df.tail(5).mean(numeric_only=True)

    display_cols = ['epoch', 'exact_match', 'true_exact', 'loss', 'tc_loss', 'magpie_loss',
                    'contrastive_loss', 'hp_loss', 'rl_loss', 'z_norm', 'entropy']

    header = f"{'Milestone':<38}"
    for c in display_cols:
        if c in df.columns:
            header += f" {c:>16}"
    print(header)
    print("-" * len(header))

    for label, row in milestones.items():
        line = f"{label:<38}"
        for c in display_cols:
            if c in df.columns:
                try:
                    val = row[c]
                except (KeyError, TypeError):
                    val = float('nan')
                if pd.isna(val):
                    line += f" {'N/A':>16}"
                elif c == 'epoch':
                    line += f" {val:>16.0f}"
                else:
                    line += f" {val:>16.6f}"
        print(line)

    print()

    # == 7. Recent Trend (last 50 epochs) ==
    print("=" * 90)
    print("RECENT TREND (last 50 epochs)")
    print("=" * 90)

    recent = df.tail(50)
    recent_epochs = recent['epoch'].values
    recent_em = recent['exact_match'].values
    recent_te = recent['true_exact'].values

    # exact_match trend
    coeffs_em = np.polyfit(recent_epochs, recent_em, 1)
    slope_em = coeffs_em[0]
    em_first10 = recent_em[:10].mean()
    em_last10 = recent_em[-10:].mean()

    print(f"\n  exact_match:")
    print(f"    First 10 of last 50 avg: {em_first10:.6f} ({em_first10*100:.2f}%)")
    print(f"    Last 10 of last 50 avg:  {em_last10:.6f} ({em_last10*100:.2f}%)")
    print(f"    Linear slope:            {slope_em:+.8f} per epoch")
    print(f"    Trend direction:         {'RISING' if slope_em > 0 else 'FALLING'}")
    print(f"    Change over 50 epochs:   {slope_em * 50:+.6f} ({slope_em * 50 * 100:+.4f}%)")

    # true_exact trend
    coeffs_te = np.polyfit(recent_epochs, recent_te, 1)
    slope_te = coeffs_te[0]
    te_first10 = recent_te[:10].mean()
    te_last10 = recent_te[-10:].mean()

    print(f"\n  true_exact:")
    print(f"    First 10 of last 50 avg: {te_first10:.6f} ({te_first10*100:.2f}%)")
    print(f"    Last 10 of last 50 avg:  {te_last10:.6f} ({te_last10*100:.2f}%)")
    print(f"    Linear slope:            {slope_te:+.8f} per epoch")
    print(f"    Trend direction:         {'RISING' if slope_te > 0 else 'FALLING'}")
    print(f"    Change over 50 epochs:   {slope_te * 50:+.6f} ({slope_te * 50 * 100:+.4f}%)")

    # loss trend
    recent_loss = recent['loss'].values
    coeffs_loss = np.polyfit(recent_epochs, recent_loss, 1)
    slope_loss = coeffs_loss[0]
    print(f"\n  loss:")
    print(f"    First 10 avg: {recent_loss[:10].mean():.6f}")
    print(f"    Last 10 avg:  {recent_loss[-10:].mean():.6f}")
    print(f"    Linear slope: {slope_loss:+.8f} per epoch")
    print(f"    Trend:        {'RISING (worsening)' if slope_loss > 0 else 'FALLING (improving)'}")

    # z_norm trend
    recent_zn = recent['z_norm'].values
    coeffs_zn = np.polyfit(recent_epochs, recent_zn, 1)
    slope_zn = coeffs_zn[0]
    print(f"\n  z_norm:")
    print(f"    First 10 avg: {recent_zn[:10].mean():.4f}")
    print(f"    Last 10 avg:  {recent_zn[-10:].mean():.4f}")
    print(f"    Linear slope: {slope_zn:+.8f} per epoch")
    print(f"    Trend:        {'RISING' if slope_zn > 0 else 'FALLING'}")

    # tc_loss trend
    recent_tc = recent['tc_loss'].values
    coeffs_tc = np.polyfit(recent_epochs, recent_tc, 1)
    slope_tc = coeffs_tc[0]
    print(f"\n  tc_loss:")
    print(f"    First 10 avg: {recent_tc[:10].mean():.6f}")
    print(f"    Last 10 avg:  {recent_tc[-10:].mean():.6f}")
    print(f"    Linear slope: {slope_tc:+.8f} per epoch")
    print(f"    Trend:        {'RISING (worsening)' if slope_tc > 0 else 'FALLING (improving)'}")

    print()

    # == 8. Epoch-by-Epoch Table (every ~25 epochs) ==
    print("=" * 90)
    print("EPOCH-BY-EPOCH TABLE (every ~25 epochs)")
    print("=" * 90)

    all_epochs = df['epoch'].values
    min_ep, max_ep = int(all_epochs.min()), int(all_epochs.max())

    target_epochs = list(range(min_ep, max_ep + 1, 25))
    if target_epochs[-1] != max_ep:
        target_epochs.append(max_ep)
    if target_epochs[0] != min_ep:
        target_epochs.insert(0, min_ep)

    selected_indices = []
    for target in target_epochs:
        closest_idx = (df['epoch'] - target).abs().idxmin()
        if closest_idx not in selected_indices:
            selected_indices.append(closest_idx)

    subset = df.loc[selected_indices]

    header = f"{'epoch':>6} {'exact_match':>12} {'true_exact':>11} {'loss':>10} {'tc_loss':>10} {'magpie_loss':>12} {'contrastive':>12} {'hp_loss':>10} {'z_norm':>8} {'entropy':>9}"
    print(header)
    print("-" * len(header))

    for _, row in subset.iterrows():
        epoch = int(row['epoch'])
        em = row['exact_match']
        te = row['true_exact']
        loss = row['loss']
        tc = row['tc_loss']
        mag = row['magpie_loss']
        cl = row['contrastive_loss'] if pd.notna(row.get('contrastive_loss')) else 0
        hp = row['hp_loss'] if pd.notna(row.get('hp_loss')) else 0
        zn = row['z_norm']
        ent = row['entropy'] if pd.notna(row.get('entropy')) else 0

        print(f"{epoch:>6} {em:>12.6f} {te:>11.6f} {loss:>10.6f} {tc:>10.6f} {mag:>12.6f} {cl:>12.6f} {hp:>10.6f} {zn:>8.2f} {ent:>9.4f}")

    print()

    # == Bonus: Correlations ==
    print("=" * 90)
    print("CORRELATIONS (across all epochs)")
    print("=" * 90)
    corr_cols = ['exact_match', 'true_exact', 'loss', 'tc_loss', 'magpie_loss',
                 'contrastive_loss', 'hp_loss', 'z_norm', 'entropy']
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()

    print(f"\n  Correlation with exact_match:")
    for c in corr_cols:
        if c != 'exact_match':
            val = corr.loc['exact_match', c]
            if pd.notna(val):
                print(f"    {c:<20}: {val:>+.4f}")

    print(f"\n  Correlation with true_exact:")
    for c in corr_cols:
        if c != 'true_exact':
            val = corr.loc['true_exact', c]
            if pd.notna(val):
                print(f"    {c:<20}: {val:>+.4f}")

    print()

    # == Bonus: Epoch ranges summary ==
    print("=" * 90)
    print("EPOCH RANGE SUMMARY (100-epoch windows)")
    print("=" * 90)

    window_size = 100
    window_targets = list(range(min_ep, max_ep, window_size))

    header = f"{'Epoch Range':<16} {'N':>4} {'exact_match':>12} {'true_exact':>11} {'loss':>10} {'contrastive':>12} {'hp_loss':>10} {'z_norm':>8}"
    print(header)
    print("-" * len(header))

    for start in window_targets:
        end = start + window_size
        window = df[(df['epoch'] >= start) & (df['epoch'] < end)]
        if len(window) == 0:
            continue
        label = f"{start}-{min(end-1, max_ep)}"
        cl_mean = window['contrastive_loss'].mean()
        hp_mean = window['hp_loss'].mean()
        cl_str = f"{cl_mean:>12.6f}" if pd.notna(cl_mean) else f"{'N/A':>12}"
        hp_str = f"{hp_mean:>10.6f}" if pd.notna(hp_mean) else f"{'N/A':>10}"
        print(f"{label:<16} {len(window):>4} {window['exact_match'].mean():>12.6f} {window['true_exact'].mean():>11.6f} "
              f"{window['loss'].mean():>10.6f} {cl_str} {hp_str} {window['z_norm'].mean():>8.2f}")

    print()

    # == Bonus: Detect major events (sudden drops/spikes) ==
    print("=" * 90)
    print("MAJOR EVENTS (sudden exact_match changes > 5%)")
    print("=" * 90)

    em_diff = df['exact_match'].diff()
    big_drops = df[em_diff < -0.05]
    big_jumps = df[em_diff > 0.05]

    if len(big_drops) > 0:
        print(f"\n  Large DROPS (>{5}%):")
        for _, row in big_drops.iterrows():
            prev_idx = row.name - 1
            if prev_idx >= 0:
                prev_em = df.loc[prev_idx, 'exact_match']
                print(f"    Epoch {int(row['epoch'])}: {prev_em:.4f} -> {row['exact_match']:.4f} "
                      f"(delta={row['exact_match']-prev_em:+.4f}, loss={row['loss']:.4f})")
    else:
        print("\n  No drops > 5% detected.")

    if len(big_jumps) > 0:
        print(f"\n  Large JUMPS (>{5}%):")
        for _, row in big_jumps.iterrows():
            prev_idx = row.name - 1
            if prev_idx >= 0:
                prev_em = df.loc[prev_idx, 'exact_match']
                print(f"    Epoch {int(row['epoch'])}: {prev_em:.4f} -> {row['exact_match']:.4f} "
                      f"(delta={row['exact_match']-prev_em:+.4f}, loss={row['loss']:.4f})")
    else:
        print("\n  No jumps > 5% detected.")

    print()

    # == Summary of key observations ==
    print("=" * 90)
    print("KEY OBSERVATIONS SUMMARY")
    print("=" * 90)
    
    # Is exact_match currently above or below peak?
    current_em = df.tail(5)['exact_match'].mean()
    peak_em = peak_em_row['exact_match']
    print(f"\n  Current exact_match (last 5 avg): {current_em:.4f} ({current_em*100:.2f}%)")
    print(f"  Peak exact_match:                 {peak_em:.4f} ({peak_em*100:.2f}%) at epoch {int(peak_em_row['epoch'])}")
    print(f"  Gap from peak:                    {current_em - peak_em:+.4f} ({(current_em - peak_em)*100:+.2f}%)")
    
    current_te = df.tail(5)['true_exact'].mean()
    peak_te = peak_te_row['true_exact']
    print(f"\n  Current true_exact (last 5 avg):  {current_te:.4f} ({current_te*100:.2f}%)")
    print(f"  Peak true_exact:                  {peak_te:.4f} ({peak_te*100:.2f}%) at epoch {int(peak_te_row['epoch'])}")
    print(f"  Gap from peak:                    {current_te - peak_te:+.4f} ({(current_te - peak_te)*100:+.2f}%)")

    # Is rl_loss / reward active?
    last_rl = df.tail(5)['rl_loss'].mean()
    last_reward = df.tail(5)['reward'].mean()
    print(f"\n  RL active? rl_loss last 5 avg:    {last_rl:.8f}")
    print(f"  Reward last 5 avg:                {last_reward:.4f}")
    
    # Entropy
    last_entropy = df.tail(5)['entropy'].mean()
    last_ew = df.tail(5)['entropy_weight'].mean()
    print(f"\n  Entropy last 5 avg:               {last_entropy:.4f}")
    print(f"  Entropy weight last 5 avg:        {last_ew:.4f}")
    
    # Teacher forcing ratio
    last_tf = df.tail(5)['tf_ratio'].mean()
    print(f"  Teacher forcing ratio last 5 avg: {last_tf:.4f}")

    print()
    print("=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
