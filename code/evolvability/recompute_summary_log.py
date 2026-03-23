#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_summary(df):
    required = {
        "p_env_site",
        "mismatch_cost",
        "init_U",
        "generation",
        "mean_U",
        "mean_parent_mismatches",
        "mean_parent_fitness",
        "expected_env_flips",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    summary = (
        df.groupby(["p_env_site", "mismatch_cost", "init_U", "generation"], as_index=False)
        .agg(
            mean_U=("mean_U", "mean"),
            sd_across_reps=("mean_U", "std"),
            mean_parent_mismatches=("mean_parent_mismatches", "mean"),
            mean_parent_fitness=("mean_parent_fitness", "mean"),
            expected_env_flips=("expected_env_flips", "mean"),
            n=("mean_U", "count"),
        )
        .sort_values(["p_env_site", "mismatch_cost", "init_U", "generation"], kind="stable")
        .reset_index(drop=True)
    )

    summary["se"] = summary["sd_across_reps"].fillna(0.0) / np.sqrt(summary["n"])
    summary["lo"] = summary["mean_U"] - 1.96 * summary["se"]
    summary["hi"] = summary["mean_U"] + 1.96 * summary["se"]

    return summary


def log_spaced_indices_from_x(x, k, preserve_until=10.0):
    """
    Choose ~k indices from a monotone x-array, preserving dense sampling for x <= preserve_until,
    then sampling approximately uniformly in log(1 + x - preserve_until) beyond that.

    Parameters
    ----------
    x : 1D array-like
        Sorted x-values, here generation.
    k : int
        Target total number of points to keep.
    preserve_until : float
        Keep all points with x <= preserve_until.
    """
    x = np.asarray(x)
    n = len(x)

    if k >= n:
        return np.arange(n, dtype=int)

    if k <= 1:
        return np.array([0], dtype=int)

    left_idx = np.where(x <= preserve_until)[0]

    # Always keep all early points if possible
    keep = set(left_idx.tolist())

    remaining_positions = np.where(x > preserve_until)[0]
    n_remaining = len(remaining_positions)

    if n_remaining == 0:
        # Too many points in the dense early region; fall back to uniform thinning there
        idx = np.linspace(0, n - 1, k)
        idx = np.round(idx).astype(int)
        return np.unique(idx)

    # How many more points can we keep after preserving the early region?
    budget_remaining = max(0, k - len(keep))

    # Always keep the last point
    keep.add(n - 1)

    # Recompute budget after forcing endpoint
    budget_remaining = max(0, k - len(keep))

    if budget_remaining == 0:
        return np.array(sorted(keep), dtype=int)

    x_tail = x[remaining_positions]
    z = np.log1p(x_tail - preserve_until)

    zmin = z[0]
    zmax = z[-1]

    if zmax == zmin:
        sampled_tail = [remaining_positions[-1]]
    else:
        targets = np.linspace(zmin, zmax, budget_remaining + 2)[1:-1]
        sampled_tail = []
        for t in targets:
            j = np.argmin(np.abs(z - t))
            sampled_tail.append(remaining_positions[j])

    keep.update(sampled_tail)

    return np.array(sorted(keep), dtype=int)


def downsample_summary(summary, max_points_per_curve=None, preserve_until=10.0):
    """
    Downsample within each (p_env_site, mismatch_cost, init_U) trajectory over generation.

    Behavior:
    - Keep nearly all points for generation <= preserve_until
    - Beyond that, keep points approximately uniformly in log-generation space
    """
    if max_points_per_curve is None:
        return summary

    pieces = []
    group_cols = ["p_env_site", "mismatch_cost", "init_U"]

    for _, g in summary.groupby(group_cols, sort=False):
        g = g.sort_values("generation", kind="stable").reset_index(drop=True)
        n = len(g)

        if n > max_points_per_curve:
            idx = log_spaced_indices_from_x(
                g["generation"].to_numpy(),
                max_points_per_curve,
                preserve_until=preserve_until,
            )
            g_small = g.iloc[idx]
        else:
            g_small = g

        pieces.append(g_small)

    out = pd.concat(pieces, axis=0, ignore_index=True)
    out = out.sort_values(
        ["p_env_site", "mismatch_cost", "init_U", "generation"],
        kind="stable",
    ).reset_index(drop=True)
    return out


def round_numeric_columns(df, decimals=None):
    if decimals is None:
        return df
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out.loc[:, num_cols] = out.loc[:, num_cols].round(decimals)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Recompute and downsample summary CSV from raw simulation CSV."
    )
    ap.add_argument("raw_csv", help="Path to raw simulation CSV")
    ap.add_argument("--out", default=None, help="Output summary CSV path")
    ap.add_argument(
        "--max-points-per-curve",
        type=int,
        default=150,
        help="Keep at most this many generation points per (p_env_site, mismatch_cost, init_U) curve",
    )
    ap.add_argument(
        "--preserve-until",
        type=float,
        default=10.0,
        help="Keep essentially all points with generation <= this value",
    )
    ap.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Round numeric columns before writing (default: 4)",
    )
    args = ap.parse_args()

    raw_csv = Path(args.raw_csv)
    df = pd.read_csv(raw_csv)

    summary = compute_summary(df)
    n_before = len(summary)

    summary = downsample_summary(
        summary,
        max_points_per_curve=args.max_points_per_curve,
        preserve_until=args.preserve_until,
    )
    summary = round_numeric_columns(summary, decimals=args.round_decimals)
    n_after = len(summary)

    if args.out is None:
        out = raw_csv.with_name(raw_csv.stem + "_summary.csv")
    else:
        out = Path(args.out)

    summary.to_csv(out, index=False)

    print(f"Wrote {out}")
    print(f"Rows: {n_before} -> {n_after}")


if __name__ == "__main__":
    main()
