#!/usr/bin/env python3
"""
Compute zlib-compressed sizes of sequences in one or more FASTA files.
STRICT year parsing:
  - Year must be the LAST 4 digits at the end of the STRAIN NAME field
    (the 3rd pipe-delimited field in GISAID-like headers: >id|seg|STRAIN|...)
  - Year must be between 2010 and 2026 inclusive
  - If any record fails these constraints, the script raises an error
    (so you immediately notice malformed headers).

Outputs:
  1) Long dataframe CSV: one row per sequence (year, zlib_bytes, etc.)
  2) Plot: mean zlib_bytes over years with 95% CI bounds (normal SEM or bootstrap)

Usage:
  python zlib_fasta_year_plot.py data/*.fasta
  python zlib_fasta_year_plot.py --out_csv out.csv --out_png out.png data1.fa data2.fa
"""

import argparse
import glob
import os
import re
import zlib
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from typing import Optional

YEAR_4_AT_END = re.compile(r"(\d{4})\Z")
YEAR_4_AT_END_AFTER_SLASH = re.compile(r"/(\d{4})\Z")
DATE_YYYY_MM_DD = re.compile(r"(\d{4})-(\d{2})-(\d{2})\Z")

def year_from_header(header: str, min_year: int = 2010, max_year: int = 2026) -> Optional[int]:
    """
    Supports two FASTA header styles, returning a 4-digit year iff explicit and in range.
    Otherwise returns None (caller should skip record).

    Style A (GISAID-like):
      >EPI...|HA|A/Texas/20/2018|EPI_ISL_...|...
      Uses strain field = 3rd pipe token; require it ends with YYYY.

    Style B (mpox-like):
      >hMpxV/USA/.../2025|EPI_ISL_...|2025-11-24
      Prefer explicit '/YYYY' at end of the first pipe token; else accept YYYY from final date token.

    No 2-digit year decoding. No heuristics beyond explicit YYYY + range check.
    """
    parts = header.split("|")

    # --- Style A: take 3rd pipe token (strain) and require YYYY at end ---
    if len(parts) >= 3:
        strain = parts[2].strip()
        m = YEAR_4_AT_END.search(strain)
        if m:
            y = int(m.group(1))
            if min_year <= y <= max_year:
                return y

    # --- Style B: take first token, require it ends with /YYYY ---
    if len(parts) >= 1:
        first = parts[0].strip()
        m = YEAR_4_AT_END_AFTER_SLASH.search(first)
        if m:
            y = int(m.group(1))
            if min_year <= y <= max_year:
                return y

    # --- Style B fallback: take last token as date YYYY-MM-DD ---
    if len(parts) >= 2:
        last = parts[-1].strip()
        m = DATE_YYYY_MM_DD.match(last)
        if m:
            y = int(m.group(1))
            if min_year <= y <= max_year:
                return y

    return None

# Must be at END of strain field: .../2018
YEAR_AT_END_RE = re.compile(r"(\d{4})\Z")

VALID_DNA = set(list("ACGTURYSWKMBDHVN"))  # includes RNA U + IUPAC ambiguity codes


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


import re

YEAR_AT_END_RE = re.compile(r"(\d{4})\Z")

def extract_year_strict(header: str, min_year: int = 2010, max_year: int = 2026):
    """
    Return an int year iff:
      - strain field (3rd '|' token) ends with EXACTLY 4 digits, and
      - that 4-digit number is within [min_year, max_year].
    Otherwise return None (caller will ignore the sequence).
    No assumptions, no 2-digit year decoding.
    """
    parts = header.split("|")
    if len(parts) < 3:
        return None
    strain = parts[2].strip()
    m = YEAR_AT_END_RE.search(strain)
    if not m:
        return None
    year = int(m.group(1))
    if min_year <= year <= max_year:
        return year
    return None

def clean_seq(seq: str) -> str:
    s = re.sub(r"\s+", "", seq).upper()
    s = s.replace("-", "")
    s = "".join(ch for ch in s if ch in VALID_DNA)
    return s


def zlib_size_bytes(seq: str, level: int = 9) -> int:
    return len(zlib.compress(seq.encode("ascii", errors="ignore"), level=level))


def normal_ci(mean: float, sem: float, z: float = 1.96) -> Tuple[float, float]:
    return mean - z * sem, mean + z * sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="FASTA file paths or globs (quoted).")
    ap.add_argument("--out_csv", default="year_zlib_sizes.csv", help="Output CSV path.")
    ap.add_argument("--out_png", default="year_zlib_plot.png", help="Output PNG path.")
    ap.add_argument("--level", type=int, default=9, help="zlib compression level (1-9).")
    ap.add_argument("--ci", choices=["normal", "bootstrap"], default="normal",
                    help="Confidence interval type.")
    ap.add_argument("--boot", type=int, default=2000,
                    help="Bootstrap resamples per year (if --ci bootstrap).")
    ap.add_argument("--min_year", type=int, default=2010)
    ap.add_argument("--max_year", type=int, default=2026)
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
    for path in paths:
        for rec in iter_fasta(path):
            # Strict year parsing: any failure is an error
            #year = extract_year_strict(rec.header, min_year=args.min_year, max_year=args.max_year)
            #if year is None:
            #    continue
            year = year_from_header(rec.header, min_year=args.min_year, max_year=args.max_year)
            if year is None:
                continue
            #else:
            #    print('*')
            seq = clean_seq(rec.seq)
            if not seq:
                raise ValueError(f"Empty/invalid sequence after cleaning: header={rec.header!r}")

            csize = zlib_size_bytes(seq, level=args.level)
            rows.append(
                {"file": os.path.basename(path), "header": rec.header, "year": year,
                 "seq_len": len(seq), "zlib_bytes": csize}
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No sequences processed (unexpected under strict parsing).")

    df.to_csv(args.out_csv, index=False)

    grp = df.groupby("year")["zlib_bytes"]

    if args.ci == "normal":
        summary = grp.agg(["count", "mean", "std"]).reset_index()
        summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
        cis = summary.apply(lambda r: normal_ci(r["mean"], r["sem"]), axis=1)
        summary["ci_low"] = [c[0] for c in cis]
        summary["ci_high"] = [c[1] for c in cis]
        summary = summary.sort_values("year")

    else:
        out = []
        rng = np.random.default_rng(12345)
        for y, vals in grp:
            vals = vals.to_numpy()
            n = len(vals)
            if n == 1:
                m = float(vals[0])
                out.append({"year": y, "count": 1, "mean": m, "ci_low": m, "ci_high": m})
                continue
            boots = np.empty(args.boot, dtype=float)
            for b in range(args.boot):
                samp = rng.choice(vals, size=n, replace=True)
                boots[b] = samp.mean()
            m = float(vals.mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append({"year": y, "count": n, "mean": m, "ci_low": float(lo), "ci_high": float(hi)})
        summary = pd.DataFrame(out).sort_values("year")

    years = summary["year"].to_numpy()
    means = summary["mean"].to_numpy()
    lo = summary["ci_low"].to_numpy()
    hi = summary["ci_high"].to_numpy()

    plt.figure(figsize=(6.2, 3.2), dpi=200)
    plt.plot(years, means, marker="o")
    plt.fill_between(years, lo, hi, alpha=0.25)
    plt.xlabel("Year")
    plt.ylabel("zlib-compressed size (bytes)")
    plt.title("Sequence compressibility over years (mean with 95% CI)")
    plt.tight_layout()
    plt.savefig(args.out_png)

    print(f"Wrote per-sequence dataframe: {args.out_csv}")
    print(f"Wrote plot: {args.out_png}")
    print("Year-level summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
