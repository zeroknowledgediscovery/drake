#!/usr/bin/env python3
"""
Compute zlib-compressed sizes of sequences in one or more FASTA files and summarize by year.

Overhead correction using a random-DNA control:
  - For each distinct sequence length n observed, generate K random A/C/G/T sequences of length n
  - Compress each with zlib at the same level
  - Estimate overhead bits O_hat(n) = median(C_rand_bits) - 2*n
  - Corrected bits/base:
      bpb_corrected = (C_real_bits - O_hat(n)) / n
    clamped to [0, 2].

Outputs:
  1) Per-sequence CSV
  2) Year summary CSV
  3) Plot (mean of chosen metric by year with 95% CI)

Naming:
  - Uses an output prefix derived from either:
      * --tag (recommended), else
      * the common prefix of input basenames (sanitized), else
      * "zlib"
  - Files written as:
      <out_dir>/<prefix>.per_sequence.csv
      <out_dir>/<prefix>.year_summary.csv
      <out_dir>/<prefix>.<plot_metric>.png

Progress:
  - Uses tqdm for FASTA parsing and overhead-control generation.

Dependencies:
  pip install tqdm

Usage:
  python zlib_fasta_year_plot.py --tag flu_HA data/*.fasta
  python zlib_fasta_year_plot.py --out_dir results --tag mpox data/*.fa
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

from tqdm import tqdm


YEAR_4_AT_END = re.compile(r"(\d{4})\Z")
YEAR_4_AT_END_AFTER_SLASH = re.compile(r"/(\d{4})\Z")
DATE_YYYY_MM_DD = re.compile(r"(\d{4})-(\d{2})-(\d{2})\Z")

VALID_DNA = set(list("ACGTURYSWKMBDHVN"))  # includes RNA U + IUPAC ambiguity codes


def year_from_header(header: str, min_year: int = 2010, max_year: int = 2026) -> Optional[int]:
    parts = header.split("|")

    if len(parts) >= 3:
        strain = parts[2].strip()
        m = YEAR_4_AT_END.search(strain)
        if m:
            y = int(m.group(1))
            if min_year <= y <= max_year:
                return y

    if len(parts) >= 1:
        first = parts[0].strip()
        m = YEAR_4_AT_END_AFTER_SLASH.search(first)
        if m:
            y = int(m.group(1))
            if min_year <= y <= max_year:
                return y

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


def count_fasta_records(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                n += 1
    return n


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
        "compression_ratio": "compression ratio (zlib_bytes / uncompressed_bytes)",
        "bytes_per_base": "compressed size (bytes/base)",
        "bits_per_base": "compressed size (bits/base)",
        "ratio_vs_2bit": "compressed bytes / 2-bit baseline bytes",
        "excess_bits_per_base": "max(0, bits/base - 2)",
        "overhead_bits_hat": "estimated overhead (bits)",
        "bits_per_base_corrected": "overhead-corrected bits/base (clamped to 0..2)",
    }
    return labels.get(metric, metric)


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def sanitize_tag(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s or "zlib"


def common_prefix_tag(paths: List[str]) -> str:
    bases = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if not bases:
        return "zlib"
    pref = os.path.commonprefix(bases)
    pref = sanitize_tag(pref)
    if len(pref) < 3:
        return "zlib"
    return pref


def estimate_overhead_by_length(lengths: List[int], level: int, k: int, seed: int) -> Dict[int, float]:
    """
    For each length n, estimate overhead bits:
      O_hat(n) = median(C_rand_bits) - 2*n
    where C_rand_bits is zlib-compressed bits of random uniform A/C/G/T strings of length n.
    Returns mapping n -> O_hat(n), clamped to >= 0.
    """
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"), dtype="<U1")

    overhead: Dict[int, float] = {}
    uniq = sorted(set(lengths))

    for n in tqdm(uniq, desc="Estimating overhead by length", unit="len"):
        c_bits = np.empty(k, dtype=float)
        for i in range(k):
            s = "".join(rng.choice(alphabet, size=n, replace=True))
            c_bytes = zlib_size_bytes_from_str(s, level=level)
            c_bits[i] = 8.0 * c_bytes
        o = float(np.median(c_bits) - 2.0 * n)
        if o < 0.0:
            o = 0.0
        overhead[n] = o

    return overhead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="FASTA file paths or globs (quoted).")

    ap.add_argument("--out_dir", default=".", help="Directory for outputs.")
    ap.add_argument("--tag", default=None,
                    help="Output prefix tag (virus/strain). If omitted, derived from input filenames.")

    ap.add_argument("--level", type=int, default=9, help="zlib compression level (1-9).")
    ap.add_argument("--ci", choices=["normal", "bootstrap"], default="normal",
                    help="Confidence interval type for the plotted metric.")
    ap.add_argument("--boot", type=int, default=2000,
                    help="Bootstrap resamples per year (if --ci bootstrap).")
    ap.add_argument("--min_year", type=int, default=2010)
    ap.add_argument("--max_year", type=int, default=2026)

    ap.add_argument("--plot_metric",
                    choices=[
                        "compression_ratio",
                        "bytes_per_base",
                        "bits_per_base",
                        "ratio_vs_2bit",
                        "excess_bits_per_base",
                        "overhead_bits_hat",
                        "bits_per_base_corrected",
                    ],
                    default="bits_per_base_corrected",
                    help="Metric to plot and compute per-year mean CI on (no concatenation).")

    # Overhead correction controls
    ap.add_argument("--overhead_k", type=int, default=50,
                    help="Random-control replicates per distinct length for overhead estimation.")
    ap.add_argument("--overhead_seed", type=int, default=123456,
                    help="RNG seed for overhead estimation.")
    ap.add_argument("--no_overhead_correction", action="store_true",
                    help="Disable overhead correction columns (faster).")
    args = ap.parse_args()

    # Expand globs
    paths: List[str] = []
    for p in args.inputs:
        expanded = glob.glob(p)
        paths.extend(expanded if expanded else [p])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("No input files found.")

    os.makedirs(args.out_dir, exist_ok=True)

    prefix = sanitize_tag(args.tag) if args.tag else common_prefix_tag(paths)

    out_csv = os.path.join(args.out_dir, f"{prefix}.per_sequence.csv")
    out_year_csv = os.path.join(args.out_dir, f"{prefix}.year_summary.csv")
    out_png = os.path.join(args.out_dir, f"{prefix}.{args.plot_metric}.png")

    # Pre-count records for progress bar (fast, but an extra scan)
    total_records = 0
    for p in paths:
        total_records += count_fasta_records(p)

    rows = []
    seq_lengths: List[int] = []

    with tqdm(total=total_records, desc="Processing FASTA records", unit="seq") as pbar:
        for path in paths:
            for rec in iter_fasta(path):
                pbar.update(1)

                year = year_from_header(rec.header, min_year=args.min_year, max_year=args.max_year)
                if year is None:
                    continue

                seq = clean_seq(rec.seq)
                if not seq:
                    raise ValueError(f"Empty/invalid sequence after cleaning: header={rec.header!r}")

                n = len(seq)
                seq_lengths.append(n)

                uncompressed_bytes = n  # ASCII bytes, 1 byte per base after cleaning
                csize = zlib_size_bytes_from_str(seq, level=args.level)

                compression_ratio = csize / uncompressed_bytes
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
                        "compression_ratio": compression_ratio,
                        "bytes_per_base": bytes_per_base,
                        "bits_per_base": bits_per_base,
                        "two_bit_bytes": two_bit_bytes,
                        "ratio_vs_2bit": ratio_vs_2bit,
                        "excess_bits_per_base": excess_bits_per_base,
                        "overhead_bits_hat": np.nan,
                        "bits_per_base_corrected": np.nan,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No sequences processed (all skipped by year parsing?).")

    # Overhead correction
    if not args.no_overhead_correction:
        overhead_map = estimate_overhead_by_length(
            lengths=seq_lengths, level=args.level, k=args.overhead_k, seed=args.overhead_seed
        )
        c_bits = 8.0 * df["zlib_bytes"].astype(float)
        n = df["seq_len"].astype(float)
        o_hat = df["seq_len"].map(overhead_map).astype(float)

        bpb_corr = (c_bits - o_hat) / n
        bpb_corr = bpb_corr.apply(lambda x: clamp(float(x), 0.0, 2.0))

        df["overhead_bits_hat"] = o_hat
        df["bits_per_base_corrected"] = bpb_corr

    df.to_csv(out_csv, index=False)

    if args.plot_metric in ("overhead_bits_hat", "bits_per_base_corrected") and args.no_overhead_correction:
        raise SystemExit("Selected plot metric requires overhead correction. Remove --no_overhead_correction.")

    summary_ci = ci_by_year(df, metric=args.plot_metric, ci_mode=args.ci, boot=args.boot)

    year_means = df.groupby("year").agg(
        n_seqs=("seq_len", "count"),
        mean_seq_len=("seq_len", "mean"),
        avg_uncompressed_bytes=("uncompressed_bytes", "mean"),
        avg_zlib_bytes=("zlib_bytes", "mean"),
        avg_compression_ratio=("compression_ratio", "mean"),
        avg_bytes_per_base=("bytes_per_base", "mean"),
        avg_bits_per_base=("bits_per_base", "mean"),
        avg_ratio_vs_2bit=("ratio_vs_2bit", "mean"),
        avg_excess_bits_per_base=("excess_bits_per_base", "mean"),
        avg_overhead_bits_hat=("overhead_bits_hat", "mean"),
        avg_bits_per_base_corrected=("bits_per_base_corrected", "mean"),
    ).reset_index()

    year_summary = summary_ci.merge(year_means, on="year", how="left").sort_values("year")
    year_summary.to_csv(out_year_csv, index=False)

    years = year_summary["year"].to_numpy()
    means = year_summary["mean"].to_numpy()
    lo = year_summary["ci_low"].to_numpy()
    hi = year_summary["ci_high"].to_numpy()

    plt.figure(figsize=(6.2, 3.2), dpi=200)
    plt.plot(years, means, marker="o")
    plt.fill_between(years, lo, hi, alpha=0.25)
    plt.xlabel("Year")
    plt.ylabel(metric_label(args.plot_metric))
    plt.title(f"{prefix}: mean {args.plot_metric} by year (95% CI)")
    plt.tight_layout()
    plt.savefig(out_png)

    print(f"Wrote per-sequence dataframe: {out_csv}")
    print(f"Wrote year-level summary: {out_year_csv}")
    print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()
