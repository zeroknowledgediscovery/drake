#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Plot panel b from saved simulation summary CSV.")
    ap.add_argument("summary_csv", help="Path to *_summary.csv file")
    ap.add_argument("--out", default=None, help="Output image path; default is alongside CSV")
    ap.add_argument("--mu-floor", type=float, default=0.00125,
                    help="Per-site mutation-rate floor used in the simulation")
    ap.add_argument("--genome-length", type=int, default=80,
                    help="Genome length n used in the simulation")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="Horizontal reference line for U")
    ap.add_argument("--title", default="b.", help="Panel title")
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv)
    df = pd.read_csv(summary_csv)

    required = {"p_env_site", "init_U", "generation", "mean_U", "lo", "hi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")

    if args.out is None:
        out = summary_csv.with_name(summary_csv.stem + "_panel_b.png")
    else:
        out = Path(args.out)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    env_vals = sorted(df["p_env_site"].unique())
    init_vals = sorted(df["init_U"].unique())

    for p_env_site in env_vals:
        label_prefix = "Static target" if p_env_site == 0 else "Changing target"
        for init_U in init_vals:
            sub = df[(df["p_env_site"] == p_env_site) & (df["init_U"] == init_U)].copy()
            sub = sub.sort_values("generation")
            if sub.empty:
                continue
            ax.plot(
                sub["generation"],
                sub["mean_U"],
                label=f"{label_prefix}, start={init_U:g}"
            )
            ax.fill_between(
                sub["generation"],
                sub["lo"],
                sub["hi"],
                alpha=0.12
            )

    ax.axhline(args.genome_length * args.mu_floor, linestyle="--", linewidth=1)
    ax.axhline(args.threshold, color="r", linestyle="--", linewidth=2, label="Threshold")
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"Mean genomic mutation intensity $U=n\mu$")
    ax.set_title(args.title)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
