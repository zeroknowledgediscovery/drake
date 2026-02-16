#!/usr/bin/env python3
"""
Compute zlib-compressed sizes of sequences in one or more FASTA files and summarize by year.

Year parsing:
  - Uses year_from_header() which supports two header styles.
  - If year is not found or not in range, the record is skipped.

Outputs:
  1) Per-sequence CSV: one row per sequence (year, zlib bytes, bytes/base, bits/base, etc.)
  2) Year summary CSV:
       - per-sequence mean of a chosen metric with 95% CI
       - concatenated-per-year compression metrics (amortizes zlib overhead)
  3) Plot: mean of chosen metric over years with 95% CI

Notes on "2 max":
  - The true entropy rate for a 4-letter alphabet is <= 2 bits/base.
  - zlib bits/base can exceed 2 due to suboptimal coding and overhead.
  - Therefore the script also reports:
      * ratio_vs_2bit  = compressed_bytes / ceil(2*bases/8)
      * excess_bits_per_base = max(0, zlib_bits_per_base - 2)

Usage:
  python zlib_fasta_year_plot.py data/*.fasta
  python zlib_fasta_year_plot.py --plot_metric bits_per_base --ci bootstrap data/*.fa
"""

import argparse
import glob
import os
import re
import zlib
import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


YEAR_4_AT_END = re.compile(r"(\d{4})\Z")
YEAR_4_AT_END_AFTER_SLASH = re.compile(r"/(\d{4})\Z")
DATE_YYYY_MM_DD = re.compile(r"(\d{4})-(\d{2})-(\d{2})\Z")

VALID_DNA = set(list("ACGTURYSWKMBDHVN"))  # includes RNA U + IUPAC ambiguity codes


def year_from_header(header: str, min_year: int = 2010, max_year: int = 2026) -> Optional[int]:
    """
    Supports two FASTA header styles, returning a 4-digit year iff explicit and in range.
    Otherwise returns None (caller skips record).

    Style A (GISAID-like):
      >EPI...|HA|A/Texas/20/2018|EPI_ISL_...|...
      Uses strain field = 3rd pipe token; require it ends with YYYY.

    Style B (mpox-like):
      >hMpxV/USA/.../2025|EPI_ISL_...|2025-11-24
      Prefer explicit '/YYYY' at end of the first pipe token; else accept YYYY from final date token.

    No 2-digit year decoding. No heuristics beyond explicit YYYY + range check.
    """
    parts = header.split("|")

    # Style A
    if len(parts) >= 3:
        strain = parts[2].strip()
        m = YEAR_4_AT_END.search(strain)
        if m:
            y = int(m.group(1))
            if min_year <= y <= max_year:
                return y

    # Style B
    if len(parts) >= 1:
        first = parts[0].strip()
        m = YEAR_4_AT_END_AFTER_SLASH.search(first)
        if m:
            y = int(m.group(1))
            if min_year <= y <= max_year:
                return y

    # Style B fallback
    if len(parts) >= 2:
        last = parts[-1].strip()
        m = DATE_YYYY_MM_DD.match(last)
        if m:
            y = int(m.group(1))
            if min_year <= y <= max_year:
                return y

    return None


@dataclass
class FastaRecord:
    header: str
    seq: str


def iter_fasta(path: str) -> Iterator[FastaRecord]:
    header = None
    chunks: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header=header, seq="".join(chunks))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield FastaRecord(header=header, seq="".join(chunks))


def clean_seq(seq: str) -> str:
    s = re.sub(r"\s+", "", seq).upper()
    s = s.replace("-", "")
    s = "".join(ch for ch in s if ch in VALID_DNA)
    return s


def zlib_size_bytes_from_str(s: str, level: int = 9) -> int:
    return len(zlib.compress(s.encode("ascii", errors="ignore"), level=level))


def normal_ci(mean: float, sem: float, z: float = 1.96) -> Tuple[float, float]:
    return mean - z * sem, mean + z * sem


def ci_by_year(df: pd.DataFrame, metric: str, ci_mode: str, boot: int) -> pd.DataFrame:
    grp = df.groupby("year")[metric]

    if ci_mode == "normal":
        summary = grp.agg(["count", "mean", "std"]).reset_index()
        summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
        cis = summary.apply(lambda r: normal_ci(r["mean"], r["sem"]), axis=1)
        summary["ci_low"] = [c[0] for c in cis]
        summary["ci_high"] = [c[1] for c in cis]
        return summary.sort_values("year")

    out = []
    rng = np.random.default_rng(12345)
    for y, vals in grp:
        vals = vals.to_numpy()
        n = len(vals)
        if n == 1:
            m = float(vals[0])
            out.append({"year": y, "count": 1, "mean": m, "ci_low": m, "ci_high": m})
            continue
        boots = np.empty(boot, dtype=float)
        for b in range(boot):
            samp = rng.choice(vals, size=n, replace=True)
            boots[b] = samp.mean()
        m = float(vals.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"year": y, "count": n, "mean": m, "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(out).sort_values("year")


def metric_label(metric: str) -> str:
    labels = {
        "zlib_bytes": "zlib-compressed size (bytes)",
        "bytes_per_base": "compressed size (bytes/base)",
        "bits_per_base": "compressed size (bits/base)",
        "excess_bits_per_base": "max(0, bits/base - 2)",
        "ratio_vs_2bit": "compressed bytes / 2-bit baseline bytes",
    }
    return labels.get(metric, metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="FASTA file paths or globs (quoted).")
    ap.add_argument("--out_csv", default="year_zlib_sizes.csv", help="Per-sequence output CSV path.")
    ap.add_argument("--out_year_csv", default="year_zlib_summary.csv", help="Year-level summary CSV path.")
    ap.add_argument("--out_png", default="year_zlib_plot.png", help="Output PNG path.")
    ap.add_argument("--level", type=int, default=9, help="zlib compression level (1-9).")
    ap.add_argument("--ci", choices=["normal", "bootstrap"], default="normal",
                    help="Confidence interval type for the plotted metric.")
    ap.add_argument("--boot", type=int, default=2000,
                    help="Bootstrap resamples per year (if --ci bootstrap).")
    ap.add_argument("--min_year", type=int, default=2010)
    ap.add_argument("--max_year", type=int, default=2026)
    ap.add_argument("--plot_metric",
                    choices=["zlib_bytes", "bytes_per_base", "bits_per_base", "excess_bits_per_base", "ratio_vs_2bit"],
                    default="bits_per_base",
                    help="Metric to plot and compute per-year mean CI on.")
    ap.add_argument("--concat_delim", default="",
                    help="Optional delimiter inserted between sequences when concatenating per year, e.g. 'N'.")
    args = ap.parse_args()

    # Expand globs
    paths: List[str] = []
    for p in args.inputs:
        expanded = glob.glob(p)
        paths.extend(expanded if expanded else [p])

    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("No input files found.")

    rows = []
    concat_by_year: Dict[int, List[str]] = {}

    for path in paths:
        for rec in iter_fasta(path):
            year = year_from_header(rec.header, min_year=args.min_year, max_year=args.max_year)
            if year is None:
                continue

            seq = clean_seq(rec.seq)
            if not seq:
                raise ValueError(f"Empty/invalid sequence after cleaning: header={rec.header!r}")

            n = len(seq)
            uncompressed_bytes = n  # ASCII bases, 1 byte per base after cleaning

            csize = zlib_size_bytes_from_str(seq, level=args.level)

            bytes_per_base = csize / n
            bits_per_base = 8.0 * csize / n

            two_bit_bytes = math.ceil(n * 2 / 8)
            ratio_vs_2bit = csize / two_bit_bytes
            excess_bits_per_base = max(0.0, bits_per_base - 2.0)

            rows.append(
                {
                    "file": os.path.basename(path),
                    "header": rec.header,
                    "year": year,
                    "seq_len": n,
                    "uncompressed_bytes": uncompressed_bytes,
                    "zlib_bytes": csize,
                    "bytes_per_base": bytes_per_base,
                    "bits_per_base": bits_per_base,
                    "two_bit_bytes": two_bit_bytes,
                    "ratio_vs_2bit": ratio_vs_2bit,
                    "excess_bits_per_base": excess_bits_per_base,
                }
            )

            concat_by_year.setdefault(year, []).append(seq)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No sequences processed.")

    df.to_csv(args.out_csv, index=False)

    # Per-year CI summary for selected metric (per-sequence)
    summary_ci = ci_by_year(df, metric=args.plot_metric, ci_mode=args.ci, boot=args.boot)

    # Year-level concatenated metrics (amortize overhead)
    y_rows = []
    for y in sorted(concat_by_year.keys()):
        if args.concat_delim:
            concat_seq = args.concat_delim.join(concat_by_year[y])
        else:
            concat_seq = "".join(concat_by_year[y])

        total_bases = len(concat_seq)
        if total_bases == 0:
            continue

        concat_zlib_bytes = zlib_size_bytes_from_str(concat_seq, level=args.level)

        concat_bytes_per_base = concat_zlib_bytes / total_bases
        concat_bits_per_base = 8.0 * concat_zlib_bytes / total_bases

        concat_two_bit_bytes = math.ceil(total_bases * 2 / 8)
        concat_ratio_vs_2bit = concat_zlib_bytes / concat_two_bit_bytes
        concat_excess_bits_per_base = max(0.0, concat_bits_per_base - 2.0)

        y_rows.append(
            {
                "year": y,
                "concat_n_seqs": len(concat_by_year[y]),
                "concat_bases": total_bases,
                "concat_zlib_bytes": concat_zlib_bytes,
                "concat_bytes_per_base": concat_bytes_per_base,
                "concat_bits_per_base": concat_bits_per_base,
                "concat_two_bit_bytes": concat_two_bit_bytes,
                "concat_ratio_vs_2bit": concat_ratio_vs_2bit,
                "concat_excess_bits_per_base": concat_excess_bits_per_base,
            }
        )
    y_df = pd.DataFrame(y_rows)

    # Add some per-year means of key per-sequence metrics (useful regardless of plot_metric)
    extra = df.groupby("year").agg(
        n_seqs=("seq_len", "count"),
        mean_seq_len=("seq_len", "mean"),
        mean_zlib_bytes=("zlib_bytes", "mean"),
        mean_bytes_per_base=("bytes_per_base", "mean"),
        mean_bits_per_base=("bits_per_base", "mean"),
        mean_ratio_vs_2bit=("ratio_vs_2bit", "mean"),
        mean_excess_bits_per_base=("excess_bits_per_base", "mean"),
    ).reset_index()

    # Merge into a single year summary table
    year_summary = summary_ci.merge(extra, on="year", how="left").merge(y_df, on="year", how="left")
    year_summary = year_summary.sort_values("year")
    year_summary.to_csv(args.out_year_csv, index=False)

    # Plot the selected metric (per-sequence mean with CI)
    years = year_summary["year"].to_numpy()
    means = year_summary["mean"].to_numpy()
    lo = year_summary["ci_low"].to_numpy()
    hi = year_summary["ci_high"].to_numpy()

    plt.figure(figsize=(6.2, 3.2), dpi=200)
    plt.plot(years, means, marker="o")
    plt.fill_between(years, lo, hi, alpha=0.25)
    plt.xlabel("Year")
    plt.ylabel(metric_label(args.plot_metric))
    plt.title(f"Mean {args.plot_metric} by year (95% CI)")
    plt.tight_layout()
    plt.savefig(args.out_png)

    print(f"Wrote per-sequence dataframe: {args.out_csv}")
    print(f"Wrote year-level summary: {args.out_year_csv}")
    print(f"Wrote plot: {args.out_png}")
    print("Year-level summary (selected columns):")
    cols = [
        "year", "count", "mean", "ci_low", "ci_high",
        "concat_n_seqs", "concat_bases",
        "concat_bits_per_base", "concat_excess_bits_per_base", "concat_ratio_vs_2bit"
    ]
    cols = [c for c in cols if c in year_summary.columns]
    print(year_summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
