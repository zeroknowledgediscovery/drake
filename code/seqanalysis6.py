#!/usr/bin/env python3
"""
Compute zlib-compressed sizes of sequences in one or more FASTA files and summarize by year.

Overhead correction using a random-DNA control:
  - For each distinct sequence length n observed (or binned length if --overhead_bin > 0),
    generate K random A/C/G/T sequences of length n
  - Compress each with zlib at the same level
  - Estimate overhead bits O_hat(n) = median(C_rand_bits) - 2*n
  - Corrected bits/base:
      bpb_corrected = (C_real_bits - O_hat(n)) / n
    clamped to [0, 2].

Outputs:
  1) Per-sequence CSV
  2) Year summary CSV
  3) Plot (mean of chosen metric by year with 95% CI)

Parallelism:
  - FASTA processing is parallelized at the file level using --numcores (default 10).
  - Overhead-control estimation is parallelized across distinct (binned) lengths using --numcores.
  - Overhead estimation is deterministic per length given --overhead_seed, independent of core count.

NCBI Datasets mode:
  Inputs are NCBI download roots or any paths under them. Each root is expected to contain:
    data_summary.tsv
    dataset_catalog.json (optional here)
    assembly_data_report.jsonl (optional here)
    <ASSEMBLY_ACCESSION>/<something>_genomic.fna

  In --ncbi mode, you can choose:
    - default (recommended): compute one row PER ASSEMBLY FILE by concatenating all contigs
      in that .fna into a single sequence (headers are kept as a semicolon-joined string).
    - --ncbi_per_record: compute one row PER FASTA RECORD (contig). This often yields many
      distinct lengths and makes overhead estimation expensive.

Dependencies:
  pip install tqdm
"""

import argparse
import glob
import os
import re
import zlib
import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Dict

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


YEAR_4_AT_END = re.compile(r"(\d{4})\Z")
YEAR_4_AT_END_AFTER_SLASH = re.compile(r"/(\d{4})\Z")
DATE_YYYY_MM_DD = re.compile(r"(\d{4})-(\d{2})-(\d{2})\Z")

VALID_DNA = set(list("ACGTURYSWKMBDHVN"))  # includes RNA U + IUPAC ambiguity codes

# NCBI assembly accession patterns
NCBI_ASM_RE = re.compile(r"\b(GCA|GCF)_\d{9}\.\d+\b")
NCBI_DATA_SUMMARY = "data_summary.tsv"


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


def find_ncbi_root(start_path: str) -> Optional[str]:
    p = os.path.abspath(start_path)
    if os.path.isfile(p):
        p = os.path.dirname(p)

    while True:
        candidate = os.path.join(p, NCBI_DATA_SUMMARY)
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


def load_ncbi_submission_dates(ncbi_root: str) -> Dict[str, str]:
    tsv_path = os.path.join(ncbi_root, NCBI_DATA_SUMMARY)
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    needed_cols = {"Assembly Accession", "Submission Date"}
    missing = needed_cols.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {tsv_path}: {sorted(missing)}")

    df = df.dropna(subset=["Assembly Accession", "Submission Date"])
    return dict(zip(df["Assembly Accession"].astype(str), df["Submission Date"].astype(str)))


def discover_ncbi_fna_files(ncbi_root: str) -> List[str]:
    pattern = os.path.join(ncbi_root, "**", "*.fna")
    files = glob.glob(pattern, recursive=True)
    return sorted([p for p in files if os.path.isfile(p)])


def ncbi_accession_from_path(fna_path: str, ncbi_root: str) -> Optional[str]:
    rel = os.path.relpath(os.path.abspath(fna_path), os.path.abspath(ncbi_root))
    parts = rel.split(os.sep)
    if parts:
        m0 = NCBI_ASM_RE.search(parts[0])
        if m0:
            return m0.group(0)
    m = NCBI_ASM_RE.search(rel)
    return m.group(0) if m else None


def year_from_iso_date(date_str: str, min_year: int, max_year: int) -> Optional[int]:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str.strip())
    if not m:
        return None
    y = int(m.group(1))
    return y if (min_year <= y <= max_year) else None


def _process_fasta_file_worker(args):
    """
    Process one FASTA file and return (rows, lengths).

    If ncbi_mode and not ncbi_per_record: concatenate all sequences in the file and produce ONE row.
    If ncbi_mode and ncbi_per_record: produce one row per FASTA record.
    Non-ncbi mode: one row per FASTA record.
    """
    (
        path,
        level,
        min_year,
        max_year,
        ncbi_mode,
        forced_year,
        ncbi_accession,
        ncbi_submission_date,
        ncbi_per_record,
    ) = args

    rows = []
    lengths = []

    if ncbi_mode and not ncbi_per_record:
        # One row per assembly file by concatenating contigs
        seq_parts: List[str] = []
        hdrs: List[str] = []
        for rec in iter_fasta(path):
            seq = clean_seq(rec.seq)
            if seq:
                seq_parts.append(seq)
                hdrs.append(rec.header)

        if not seq_parts:
            return [], []

        year = forced_year
        if year is None:
            return [], []

        seq_all = "".join(seq_parts)
        n = len(seq_all)
        csize = zlib_size_bytes_from_str(seq_all, level=level)

        compression_ratio = csize / n
        bytes_per_base = csize / n
        bits_per_base = 8.0 * csize / n

        two_bit_bytes = math.ceil(n * 2 / 8)
        ratio_vs_2bit = csize / two_bit_bytes
        excess_bits_per_base = max(0.0, bits_per_base - 2.0)

        # Store a compact header: first header + count
        header_out = hdrs[0]
        if len(hdrs) > 1:
            header_out = f"{hdrs[0]} ;(+{len(hdrs)-1} contigs)"

        rows.append(
            {
                "file": os.path.basename(path),
                "header": header_out,
                "year": year,
                "seq_len": n,
                "uncompressed_bytes": n,
                "zlib_bytes": csize,
                "compression_ratio": compression_ratio,
                "bytes_per_base": bytes_per_base,
                "bits_per_base": bits_per_base,
                "two_bit_bytes": two_bit_bytes,
                "ratio_vs_2bit": ratio_vs_2bit,
                "excess_bits_per_base": excess_bits_per_base,
                "overhead_bits_hat": np.nan,
                "bits_per_base_corrected": np.nan,
                "ncbi_accession": ncbi_accession,
                "ncbi_submission_date": ncbi_submission_date,
                "ncbi_n_contigs": len(hdrs),
            }
        )
        lengths.append(n)
        return rows, lengths

    # Otherwise: per FASTA record
    for rec in iter_fasta(path):
        if ncbi_mode:
            year = forced_year
        else:
            year = year_from_header(rec.header, min_year=min_year, max_year=max_year)

        if year is None:
            continue

        seq = clean_seq(rec.seq)
        if not seq:
            continue

        n = len(seq)
        lengths.append(n)

        csize = zlib_size_bytes_from_str(seq, level=level)

        compression_ratio = csize / n
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
                "uncompressed_bytes": n,
                "zlib_bytes": csize,
                "compression_ratio": compression_ratio,
                "bytes_per_base": bytes_per_base,
                "bits_per_base": bits_per_base,
                "two_bit_bytes": two_bit_bytes,
                "ratio_vs_2bit": ratio_vs_2bit,
                "excess_bits_per_base": excess_bits_per_base,
                "overhead_bits_hat": np.nan,
                "bits_per_base_corrected": np.nan,
                "ncbi_accession": ncbi_accession if ncbi_mode else np.nan,
                "ncbi_submission_date": ncbi_submission_date if ncbi_mode else np.nan,
                "ncbi_n_contigs": np.nan,
            }
        )

    return rows, lengths


def _overhead_worker(args):
    n, level, k, seed = args
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"), dtype="<U1")
    c_bits = np.empty(k, dtype=float)
    for i in range(k):
        s = "".join(rng.choice(alphabet, size=n, replace=True))
        c_bytes = zlib_size_bytes_from_str(s, level=level)
        c_bits[i] = 8.0 * c_bytes
    o = float(np.median(c_bits) - 2.0 * n)
    if o < 0.0:
        o = 0.0
    return n, o


def estimate_overhead_by_length_parallel(
    lengths: List[int], level: int, k: int, seed: int, numcores: int
) -> Dict[int, float]:
    uniq = sorted(set(lengths))
    if not uniq:
        return {}

    tasks = []
    for n in uniq:
        per_seed = (seed + (n * 1000003)) & 0xFFFFFFFF
        tasks.append((n, level, k, per_seed))

    overhead: Dict[int, float] = {}
    with ProcessPoolExecutor(max_workers=numcores) as ex:
        futs = [ex.submit(_overhead_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Estimating overhead by length", unit="len"):
            n, o = fut.result()
            overhead[n] = o

    return overhead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="FASTA file paths or globs (quoted), or NCBI root dirs when --ncbi.")

    ap.add_argument("--out_dir", default=".", help="Directory for outputs.")
    ap.add_argument("--tag", default=None, help="Output prefix tag (virus/strain).")

    ap.add_argument("--level", type=int, default=9, help="zlib compression level (1-9).")
    ap.add_argument("--ci", choices=["normal", "bootstrap"], default="normal")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min_year", type=int, default=2010)
    ap.add_argument("--max_year", type=int, default=2026)

    ap.add_argument(
        "--plot_metric",
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
    )

    ap.add_argument("--overhead_k", type=int, default=50)
    ap.add_argument("--overhead_seed", type=int, default=123456)
    ap.add_argument("--no_overhead_correction", action="store_true")

    ap.add_argument("--ncbi", action="store_true")
    ap.add_argument(
        "--ncbi_per_record",
        action="store_true",
        help="In --ncbi mode, compute per FASTA record (contig) instead of per assembly file (concatenated).",
    )

    ap.add_argument("--numcores", type=int, default=10)

    ap.add_argument(
        "--overhead_bin",
        type=int,
        default=0,
        help="If >0, bin sequence lengths to nearest multiple of this before overhead estimation (e.g., 1000).",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_paths: List[str] = []
    for p in args.inputs:
        expanded = glob.glob(p)
        raw_paths.extend(expanded if expanded else [p])
    raw_paths = [p for p in raw_paths if os.path.exists(p)]
    if not raw_paths:
        raise SystemExit("No input files found.")

    ncbi_root_to_dates: Dict[str, Dict[str, str]] = {}
    fpaths: List[str] = []

    if args.ncbi:
        roots = set()
        for p in raw_paths:
            root = find_ncbi_root(p)
            if root is None:
                raise SystemExit(f"--ncbi set but could not find {NCBI_DATA_SUMMARY} above: {p}")
            roots.add(root)

        for root in sorted(roots):
            if root not in ncbi_root_to_dates:
                ncbi_root_to_dates[root] = load_ncbi_submission_dates(root)
            fpaths.extend(discover_ncbi_fna_files(root))

        if not fpaths:
            raise SystemExit("No .fna files discovered under the provided NCBI roots.")

        prefix = sanitize_tag(args.tag) if args.tag else "zlib"
    else:
        paths: List[str] = []
        for p in raw_paths:
            if os.path.isdir(p):
                for ext in ("*.fa", "*.fasta", "*.fna", "*.fas"):
                    paths.extend(glob.glob(os.path.join(p, ext)))
            else:
                paths.append(p)

        fpaths = [p for p in paths if os.path.isfile(p)]
        if not fpaths:
            raise SystemExit("No input FASTA files found.")
        prefix = sanitize_tag(args.tag) if args.tag else common_prefix_tag(fpaths)

    out_csv = os.path.join(args.out_dir, f"{prefix}.per_sequence.csv")
    out_year_csv = os.path.join(args.out_dir, f"{prefix}.year_summary.csv")
    out_png = os.path.join(args.out_dir, f"{prefix}.{args.plot_metric}.png")

    tasks = []
    for path in fpaths:
        ncbi_accession = None
        ncbi_submission_date = None
        forced_year = None

        if args.ncbi:
            root = find_ncbi_root(path)
            if root is None:
                continue
            dates_map = ncbi_root_to_dates.get(root, {})
            ncbi_accession = ncbi_accession_from_path(path, root)
            if ncbi_accession is not None:
                ncbi_submission_date = dates_map.get(ncbi_accession)
                if ncbi_submission_date is not None:
                    forced_year = year_from_iso_date(ncbi_submission_date, args.min_year, args.max_year)

        tasks.append(
            (
                path,
                args.level,
                args.min_year,
                args.max_year,
                args.ncbi,
                forced_year,
                ncbi_accession,
                ncbi_submission_date,
                args.ncbi_per_record,
            )
        )

    rows = []
    seq_lengths: List[int] = []

    with ProcessPoolExecutor(max_workers=args.numcores) as ex:
        futs = [ex.submit(_process_fasta_file_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing FASTA files", unit="file"):
            r, lens = fut.result()
            if r:
                rows.extend(r)
            if lens:
                seq_lengths.extend(lens)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No sequences processed (all skipped by year parsing?).")

    if not args.no_overhead_correction:
        if args.overhead_bin and args.overhead_bin > 0:
            seq_lengths_for_overhead = [int(round(n / args.overhead_bin) * args.overhead_bin) for n in seq_lengths]
        else:
            seq_lengths_for_overhead = seq_lengths

        overhead_map = estimate_overhead_by_length_parallel(
            lengths=seq_lengths_for_overhead,
            level=args.level,
            k=args.overhead_k,
            seed=args.overhead_seed,
            numcores=args.numcores,
        )

        c_bits = 8.0 * df["zlib_bytes"].astype(float)
        n = df["seq_len"].astype(float)

        if args.overhead_bin and args.overhead_bin > 0:
            len_key = (df["seq_len"] / args.overhead_bin).round() * args.overhead_bin
            len_key = len_key.astype(int)
        else:
            len_key = df["seq_len"].astype(int)

        o_hat = len_key.map(overhead_map).astype(float)

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
