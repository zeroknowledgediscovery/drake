
# Drake Project – Python Scripts (code/pycode)

This directory contains simulation and genome-compression utilities used in the Drake manuscript and associated supplementary analyses. Each script is standalone and executable from the command line.

---

## 1. sim.py

Toy temperature-scaling simulation for the mutation–discovery model.

Purpose:
Implements the theoretical discovery-rate function Φ(μ) and compares it to a Monte Carlo proxy. Produces scaling plots in terms of both per-site mutation rate μ and mutation “temperature” T = nμ.

Core ideas:
- Exact theoretical expression for Φ(μ)
- Monte Carlo binomial sampling of mutation counts
- Demonstrates collapse onto T e^{-T} scaling

Outputs:
- temperature_scaling_sim.png
- temperature_scaling_sim.pdf
- temperature_scaling_sim_data.csv

No CLI arguments; parameters are defined at the top of the script.

Run:
    python sim.py

---

## 2. noiseamp2.py

Toy entropy-drift simulation (SI Fig S1 panel).

Purpose:
Simulates sequence evolution under:
(A) mutation only
(B) mutation with viability constraint

Measures empirical bigram conditional entropy H(X_t | X_{t-1}) over generations.

Outputs:
- figS1_toy_entropy_proxy.png
- figS1_toy_entropy_proxy.pdf
- figS1_toy_entropy_proxy_plotdata.csv

Parameters (editable at top):
- n: sequence length
- G: generations
- mu: per-site mutation rate
- reps: replicate lineages
- MAX_MISMATCH_FRAC: viability band

Run:
    python noiseamp2.py

---

## 3. seqanalysis8.py

NCBI assembly-level genome compression analysis.

Purpose:
Processes NCBI Datasets genome downloads and computes:

- genome length (n_bases)
- optimal 2-bit baseline (n/4)
- zlib compressed size
- additive-overhead–corrected zlib size

Maps assemblies to submission year using data_summary.tsv.

Inputs:
NCBI dataset root containing:
    data_summary.tsv
    GCA_*/ or GCF_*/ directories with .fna files

Example:
    python seqanalysis8.py --root ./ncbi_dataset/data --tag salmonella --numcores 10

Outputs:
- <tag>.per_assembly.csv
- <tag>.year_summary.csv

---

## 4. seqanalysis4.py

Per-sequence FASTA compression analysis and year-level plotting.

Purpose:
Computes zlib compression metrics for arbitrary FASTA files or NCBI datasets.

Metrics include:
- bits_per_base
- ratio_vs_2bit
- excess_bits_per_base
- overhead_bits_hat
- bits_per_base_corrected (default plotted metric)

Supports:
- Normal or bootstrap confidence intervals
- Overhead correction via random-DNA control
- NCBI Datasets mode

Examples:

General FASTA:
    python seqanalysis4.py --tag flu data/*.fasta

NCBI mode:
    python seqanalysis4.py --ncbi --tag salmonella /path/to/ncbi_root

Outputs:
- <prefix>.per_sequence.csv
- <prefix>.year_summary.csv
- <prefix>.<plot_metric>.png

Dependencies:
    pip install tqdm pandas numpy matplotlib

---

Reproducibility Notes:

- All scripts are deterministic unless Monte Carlo sampling is used.
- RNG seeds are defined at the top of simulation scripts.
- zlib level defaults to 9.
