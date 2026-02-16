#!/usr/bin/env python3
"""
NCBI Datasets end-to-end downloader:
1) Query a taxon (organism name) with `datasets summary genome taxon`
2) Flatten metadata with `dataformat tsv genome`
3) Filter to years [YMIN, YMAX] using collection-date if present, else release-date, else submission-date
4) Cap to MAXN assemblies (deterministic random sample)
5) Download as dehydrated package + rehydrate
6) Concatenate all *.fna into one FASTA, rewriting headers to include year:
   >ASSEMBLY_ACCESSION|YYYY|original_header

Requirements:
- NCBI Datasets CLI installed and in PATH: `datasets`, `dataformat`
  Docs: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/datasets/
"""

import argparse
import csv
import glob
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATE_FIELDS = ["collection-date", "release-date", "submission-date"]


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )


def which_or_die(exe: str) -> None:
    if shutil.which(exe) is None:
        raise RuntimeError(f"Missing required executable in PATH: {exe}")


def parse_year(s: str) -> Optional[int]:
    if not s or s.strip() in {"-", ""}:
        return None
    m = re.match(r"^(\d{4})", s.strip())
    return int(m.group(1)) if m else None


def try_dataformat_fields(meta_json: Path, out_tsv: Path, fields: List[str]) -> bool:
    cmd = ["dataformat", "tsv", "genome", "--inputfile", str(meta_json), "--fields", ",".join(fields)]
    try:
        # dataformat writes to stdout; capture by redirecting via python
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return False
        out_tsv.write_text(p.stdout)
        return True
    except Exception:
        return False


def load_meta_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def select_accessions(
    rows: List[Dict[str, str]],
    ymin: int,
    ymax: int,
    maxn: int,
    seed: int,
    require_complete: bool,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns: selected accessions, year_by_accession (YYYY string)
    """
    eligible: List[Tuple[str, int]] = []
    year_by_acc: Dict[str, str] = {}

    for r in rows:
        acc = r.get("accession", "").strip()
        if not acc:
            continue

        if require_complete:
            lvl = (r.get("assembly-level") or r.get("assembly_level") or "").strip().lower()
            # dataformat field name is usually "assembly-level"
            if "complete" not in lvl:
                continue

        y = None
        for f in DATE_FIELDS:
            y = parse_year(r.get(f, ""))
            if y is not None:
                break
        if y is None:
            continue
        if ymin <= y <= ymax:
            eligible.append((acc, y))
            year_by_acc[acc] = str(y)

    eligible.sort(key=lambda t: t[0])

    if len(eligible) > maxn:
        random.seed(seed)
        sample = random.sample(eligible, maxn)
        sample.sort(key=lambda t: t[0])
        selected = [a for a, _ in sample]
    else:
        selected = [a for a, _ in eligible]

    return selected, year_by_acc


def write_accessions(accessions: List[str], path: Path) -> None:
    path.write_text("".join(a + "\n" for a in accessions))


def concatenate_fasta_with_dates(
    pkg_data_dir: Path,
    out_fasta: Path,
    year_by_acc: Dict[str, str],
) -> int:
    fna_files = glob.glob(str(pkg_data_dir / "*" / "*.fna"))
    fna_files.sort()
    n_files = 0

    with out_fasta.open("w") as out:
        for fp in fna_files:
            fp_path = Path(fp)
            acc = fp_path.parent.name
            year = year_by_acc.get(acc, "NA")

            with fp_path.open("r") as f:
                for line in f:
                    if line.startswith(">"):
                        hdr = line[1:].strip()
                        out.write(f">{acc}|{year}|{hdr}\n")
                    else:
                        out.write(line)
            n_files += 1

    return n_files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=True, help='Taxon name, e.g. "Salmonella enterica" or "Escherichia coli"')
    ap.add_argument("--ymin", type=int, default=2025)
    ap.add_argument("--ymax", type=int, default=2026)
    ap.add_argument("--maxn", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0, help="Deterministic sampling seed if more than maxn assemblies")
    ap.add_argument("--require-complete", action="store_true", help="Keep only assemblies with Complete Genome level")
    ap.add_argument("--workdir", default="ncbi_pull", help="Working directory")
    ap.add_argument("--out-fasta", default="combined_with_dates.fasta", help="Output FASTA path")
    args = ap.parse_args()

    which_or_die("datasets")
    which_or_die("dataformat")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    meta_json = workdir / "meta.json"
    meta_tsv = workdir / "meta.tsv"
    acc_list = workdir / "accessions.txt"
    dl_dir = workdir / "dl"
    pkg_dir = dl_dir / "pkg"
    out_fasta = Path(args.out_fasta).resolve()

    print(f"[1/6] Fetching genome summary JSON for: {args.organism}")
    # datasets summary writes to stdout, so capture via subprocess and write file
    p = subprocess.run(
        ["datasets", "summary", "genome", "taxon", args.organism],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"datasets summary failed:\n{p.stderr}")
    meta_json.write_text(p.stdout)

    print("[2/6] Flattening metadata to TSV")
    # Try richer fields first; if collection-date unsupported, fall back.
    rich_fields = [
        "accession",
        "organism-name",
        "assembly-level",
        "assembly-source",
        "release-date",
        "submission-date",
        "collection-date",
        "biosample-accession",
        "bioproject-accession",
        "submitter",
    ]
    ok = try_dataformat_fields(meta_json, meta_tsv, rich_fields)
    if not ok:
        fallback_fields = [
            "accession",
            "organism-name",
            "assembly-level",
            "assembly-source",
            "release-date",
            "submission-date",
            "biosample-accession",
            "bioproject-accession",
            "submitter",
        ]
        ok2 = try_dataformat_fields(meta_json, meta_tsv, fallback_fields)
        if not ok2:
            raise RuntimeError("dataformat failed for both rich and fallback field sets. Check your Datasets CLI version.")
        else:
            print("  Note: collection-date field not available; will use release-date/submission-date.")

    rows = load_meta_tsv(meta_tsv)

    print(f"[3/6] Filtering to years {args.ymin}..{args.ymax} and capping at {args.maxn}")
    selected, year_by_acc = select_accessions(
        rows=rows,
        ymin=args.ymin,
        ymax=args.ymax,
        maxn=args.maxn,
        seed=args.seed,
        require_complete=args.require_complete,
    )
    if not selected:
        raise RuntimeError("No assemblies matched your filters. Try turning off --require-complete or widen years.")

    write_accessions(selected, acc_list)
    print(f"  Selected assemblies: {len(selected)}")
    print(f"  Accession list: {acc_list}")

    print("[4/6] Downloading dehydrated genome package (this may be large)")
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / "genomes.zip"
    run(
        [
            "datasets",
            "download",
            "genome",
            "accession",
            "--inputfile",
            str(acc_list),
            "--dehydrated",
            "--include",
            "genome",
            "--filename",
            str(zip_path),
        ]
    )

    print("[5/6] Unzipping and rehydrating package")
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    run(["unzip", "-q", str(zip_path), "-d", str(pkg_dir)])
    run(["datasets", "rehydrate", "--directory", str(pkg_dir)])

    data_dir = pkg_dir / "ncbi_dataset" / "data"
    if not data_dir.exists():
        raise RuntimeError(f"Expected data directory not found: {data_dir}")

    print("[6/6] Concatenating FASTA files and rewriting headers with year")
    n_files = concatenate_fasta_with_dates(data_dir, out_fasta, year_by_acc)
    print(f"  Wrote: {out_fasta}")
    print(f"  Source FASTA files: {n_files}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
