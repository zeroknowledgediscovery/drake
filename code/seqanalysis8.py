#!/usr/bin/env python3
"""
NCBI-assembly genome compressibility (minimal script)

Input directory layout (NCBI Datasets genome download):
  <ROOT>/
    data_summary.tsv
    assembly_data_report.jsonl (ignored)
    dataset_catalog.json (ignored)
    GCA_.../
      *.fna

What this script does:
  1) Finds all .fna files under ROOT/*/*.fna
  2) Maps each assembly accession (GCA_... or GCF_...) to year using data_summary.tsv "Submission Date"
  3) For each .fna file, concatenates contigs into one genome string (A/C/G/T only by default)
  4) Computes:
       - n = genome length (bases)
       - optimal_2bit_bytes = n/4
       - zlib_bytes = len(zlib.compress(ascii(genome)))
       - zlib_bytes_corrected = zlib_bytes - const_hat/8
         where const_hat is estimated once from random DNA at length n0:
             const_hat = median( C_rand_bits(n0) - 8*n0 )
         (this estimates zlib's additive constant overhead in bits)
  5) Writes a per-assembly CSV and a simple year summary CSV.

Notes:
  - "optimal_2bit_bytes" is the 2-bit encoding baseline (Shannon ceiling for uniform A/C/G/T).
  - zlib works on ASCII (8 bits/base), so zlib_bytes is usually > optimal_2bit_bytes.
  - The correction here removes only an additive constant (format/block overhead), not the 6n slope.
    That is the right correction if you want to compare zlib size against the 2-bit baseline.

Example:
  python ncbi_zlib_min.py --root ./ncbi_dataset/data --tag salmonella --numcores 10
"""

import argparse
import os
import re
import math
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

VALID_ACGT = set("ACGT")
ASM_RE = re.compile(r"\b(GCA|GCF)_\d{9}\.\d+\b")


def iter_fasta_sequences(path: str):
    """Yield (header, seq_string) from a FASTA file."""
    header = None
    chunks = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def clean_to_acgt(seq: str) -> str:
    s = re.sub(r"\s+", "", seq).upper().replace("-", "")
    # keep only A/C/G/T
    return "".join(ch for ch in s if ch in VALID_ACGT)


def zlib_bytes_ascii(s: str, level: int) -> int:
    return len(zlib.compress(s.encode("ascii", errors="ignore"), level=level))


def discover_fna_files(root: str):
    out = []
    for name in os.listdir(root):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        # one level deep: GCA_.../*.fna
        for fn in os.listdir(d):
            if fn.endswith(".fna"):
                out.append(os.path.join(d, fn))
    return sorted(out)


def accession_from_path(path: str) -> str | None:
    # Prefer directory name
    d = os.path.basename(os.path.dirname(path))
    m = ASM_RE.search(d)
    if m:
        return m.group(0)
    m = ASM_RE.search(path)
    return m.group(0) if m else None


def load_year_map(data_summary_tsv: str) -> dict[str, int]:
    df = pd.read_csv(data_summary_tsv, sep="\t", dtype=str, low_memory=False)
    needed = {"Assembly Accession", "Submission Date"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"data_summary.tsv missing columns: {sorted(missing)}")

    out = {}
    for acc, date in zip(df["Assembly Accession"].astype(str), df["Submission Date"].astype(str)):
        if not isinstance(date, str):
            continue
        m = re.match(r"^(\d{4})-\d{2}-\d{2}$", date.strip())
        if not m:
            continue
        out[acc] = int(m.group(1))
    return out






def estimate_const2_hat_bits(level: int, n0: int, k: int, seed: int) -> float:
    """
    Estimate additive constant in zlib bits for random A/C/G/T strings relative to 2 bits/base:
      C_rand_bits(n) ≈ 2n + const2
    const2_hat = median(C_rand_bits(n0) - 2n0).
    """
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"), dtype="<U1")
    kk = int(max(1, k))
    c_bits = np.empty(kk, dtype=float)
    for i in range(kk):
        s = "".join(rng.choice(alphabet, size=n0, replace=True))
        c_bits[i] = 8.0 * zlib_bytes_ascii(s, level=level)
    return float(np.median(c_bits - 2.0 * n0))











def estimate_const_hat_bits(level: int, n0: int, k: int, seed: int) -> float:
    """
    Estimate additive constant in zlib bits for random A/C/G/T ASCII strings:
      C_rand_bits(n) ≈ 8n + const
    const_hat = median(C_rand_bits(n0) - 8n0).
    """
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"), dtype="<U1")
    kk = int(max(1, k))
    c_bits = np.empty(kk, dtype=float)
    for i in range(kk):
        s = "".join(rng.choice(alphabet, size=n0, replace=True))
        c_bits[i] = 8.0 * zlib_bytes_ascii(s, level=level)
    return float(np.median(c_bits - 8.0 * n0))


def process_one_fna(args):
    path, level = args
    acc = accession_from_path(path)

    seq_parts = []
    n_contigs = 0
    for _, seq in iter_fasta_sequences(path):
        s = clean_to_acgt(seq)
        if s:
            seq_parts.append(s)
            n_contigs += 1

    if not seq_parts:
        return None

    genome = "".join(seq_parts)
    n = len(genome)
    zbytes = zlib_bytes_ascii(genome, level=level)
    return {
        "accession": acc if acc is not None else "",
        "file": os.path.basename(path),
        "path": path,
        "n_contigs": n_contigs,
        "n_bases": n,
        "optimal_2bit_bytes": n / 4.0,
        "zlib_bytes": int(zbytes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="NCBI dataset root containing data_summary.tsv and GCA_*/ dirs")
    ap.add_argument("--tag", default="ncbi", help="Output prefix tag")
    ap.add_argument("--out_dir", default=".", help="Output directory")
    ap.add_argument("--level", type=int, default=9, help="zlib level 1..9")
    ap.add_argument("--numcores", type=int, default=10, help="Parallel workers")
    # constant-overhead estimate
    ap.add_argument("--n0", type=int, default=200000, help="Random DNA length for const estimation")
    ap.add_argument("--k0", type=int, default=10, help="Replicates for const estimation (10 is enough)")
    ap.add_argument("--seed", type=int, default=123456, help="RNG seed for const estimation")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(args.out_dir, exist_ok=True)

    data_summary = os.path.join(root, "data_summary.tsv")
    if not os.path.isfile(data_summary):
        raise SystemExit(f"Missing {data_summary}")

    year_map = load_year_map(data_summary)

    fna_files = discover_fna_files(root)
    if not fna_files:
        raise SystemExit(f"No .fna files found under {root}")

    # Estimate additive constant once
    #const_hat_bits = estimate_const_hat_bits(level=args.level, n0=args.n0, k=args.k0, seed=args.seed)
    #const_hat_bytes = const_hat_bits / 8.0


    const2_hat_bits = estimate_const2_hat_bits(level=args.level, n0=args.n0, k=args.k0, seed=args.seed)
    const2_hat_bytes = const2_hat_bits / 8.0

    

    rows = []
    with ProcessPoolExecutor(max_workers=args.numcores) as ex:
        futs = [ex.submit(process_one_fna, (p, args.level)) for p in fna_files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing .fna", unit="file"):
            r = fut.result()
            if r is None:
                continue
            acc = r["accession"]
            r["year"] = year_map.get(acc, np.nan)
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No genomes processed (all empty after cleaning?)")

    # Correct only the additive constant (bytes)
    #df["zlib_bytes_corrected"] = (df["zlib_bytes"].astype(float) - const_hat_bytes).clip(lower=0.0)
    df["zlib_bytes_corrected"] = (df["zlib_bytes"].astype(float) - const2_hat_bytes).clip(lower=0.0)
    # Keep exactly the columns you asked for (plus accession/year, which you will want)
    df_out = df[
        [
            "accession",
            "year",
            "n_bases",
            "optimal_2bit_bytes",
            "zlib_bytes",
            "zlib_bytes_corrected",
            "n_contigs",
            "file",
        ]
    ].sort_values(["year", "accession"], na_position="last")

    out_csv = os.path.join(args.out_dir, f"{args.tag}.per_assembly.csv")
    df_out.to_csv(out_csv, index=False)

    # Simple year summary (means)
    year_summary = (
        df_out.dropna(subset=["year"])
        .groupby("year")
        .agg(
            n_assemblies=("accession", "count"),
            mean_n_bases=("n_bases", "mean"),
            mean_optimal_2bit_bytes=("optimal_2bit_bytes", "mean"),
            mean_zlib_bytes=("zlib_bytes", "mean"),
            mean_zlib_bytes_corrected=("zlib_bytes_corrected", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )
    out_year = os.path.join(args.out_dir, f"{args.tag}.year_summary.csv")
    year_summary.to_csv(out_year, index=False)

    print(f"const2_hat_bits={const2_hat_bits:.2f}  const2_hat_bytes={const2_hat_bytes:.2f}")
    #print(f"const_hat_bits={const_hat_bits:.2f}  const_hat_bytes={const_hat_bytes:.2f}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_year}")


if __name__ == "__main__":
    main()
