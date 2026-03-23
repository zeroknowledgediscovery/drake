#!/usr/bin/env python3
import os
import json
import uuid
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
from datetime import datetime, timezone


def evolve_population(
    N=400,  # population size
    n=1000,  # genome length
    G=100000,
    init_U=0.05,
    mu_floor=0.00125,  # U_floor = 0.1 when n=80
    mu_cap=0.20,
    modifier_mut_prob=0.05,
    log_step_sd=0.18,
    mismatch_cost=0.28,
    p_env_site=0.0,  # per-site probability target flips each generation
    offspring_pool_factor=2,
    seed=0,
):
    rng = np.random.default_rng(seed)
    genomes = np.zeros((N, n), dtype=np.bool_)
    target = np.zeros(n, dtype=np.bool_)
    mu = np.full(N, init_U / n, dtype=float)

    rows = []
    for g in range(G + 1):
        parent_mismatches = np.count_nonzero(genomes != target, axis=1)
        parent_fitness = np.exp(-mismatch_cost * parent_mismatches)

        rows.append({
            "generation": g,
            "mean_U": float(n * mu.mean()),
            "median_U": float(n * np.median(mu)),
            "sd_U": float(n * mu.std(ddof=1)),
            "mean_parent_mismatches": float(parent_mismatches.mean()),
            "mean_parent_fitness": float(parent_fitness.mean()),
            "p_env_site": p_env_site,
            "expected_env_flips": float(n * p_env_site),
            "init_U": init_U,
            "mismatch_cost": mismatch_cost,
            "seed": seed,
        })

        if g == G:
            break

        # Parent selection
        p_parent = parent_fitness + 1e-16
        p_parent /= p_parent.sum()
        M = offspring_pool_factor * N
        parent_idx = rng.choice(N, size=M, replace=True, p=p_parent)

        offspring_genomes = genomes[parent_idx].copy()
        offspring_mu = mu[parent_idx].copy()

        # Sequence mutation
        mut_mask = rng.random((M, n)) < offspring_mu[:, None]
        offspring_genomes ^= mut_mask

        # Modifier mutations
        mod_mask = rng.random(M) < modifier_mut_prob
        if mod_mask.any():
            offspring_mu[mod_mask] *= np.exp(rng.normal(0.0, log_step_sd, size=mod_mask.sum()))
        offspring_mu = np.clip(offspring_mu, mu_floor, mu_cap)

        # Offspring viability
        offspring_mismatches = np.count_nonzero(offspring_genomes != target, axis=1)
        offspring_fitness = np.exp(-mismatch_cost * offspring_mismatches) + 1e-16
        p_offspring = offspring_fitness / offspring_fitness.sum()

        survivors = rng.choice(M, size=N, replace=True, p=p_offspring)
        genomes = offspring_genomes[survivors]
        mu = offspring_mu[survivors]

        # Environmental change: each target site flips independently with probability p_env_site
        if p_env_site > 0:
            env_mask = rng.random(n) < p_env_site
            target ^= env_mask

    return pd.DataFrame(rows)


def _run_one(task):
    return evolve_population(**task)


def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel explicit-genome mutation-modifier simulation with offspring viability and per-site environmental change."
    )
    p.add_argument("--outdir", type=str, default="./", help="Output directory")
    p.add_argument("--reps", type=int, default=16, help="Replicates per (environment, initial U)")
    p.add_argument("--generations", type=int, default=10000, help="Number of generations")
    p.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8), help="Number of worker processes")
    p.add_argument("--population", type=int, default=400, help="Population size N")
    p.add_argument("--genome-length", type=int, default=80, help="Genome length n")
    p.add_argument("--mu-floor", type=float, default=0.00125, help="Lower floor on per-site mutation rate")
    p.add_argument("--mu-cap", type=float, default=0.20, help="Upper cap on per-site mutation rate")
    p.add_argument("--modifier-mut-prob", type=float, default=0.05, help="Probability of modifier mutation per offspring")
    p.add_argument("--log-step-sd", type=float, default=0.18, help="SD of multiplicative modifier step in log space")
    p.add_argument("--mismatch-cost", type=float, default=0.28, help="Fitness penalty per target mismatch")
    p.add_argument("--offspring-pool-factor", type=int, default=2, help="Offspring pool size as multiple of N")
    p.add_argument("--starts", type=float, nargs="+", default=[0.05, 0.20, 1.50, 3.00],
                   help="Initial genomic mutation intensities U=n*mu")
    p.add_argument("--env-site-probs", type=float, nargs="+", default=[0.0, 0.0125],
                   help="Per-site target-flip probabilities per generation")
    p.add_argument("--seed-base", type=int, default=None, help="Base seed")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Optional custom tag to include in all output filenames")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    return p.parse_args()


def make_run_tag(user_tag=None):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = uuid.uuid4().hex[:8]
    if user_tag:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in user_tag)
        return f"{safe}_{ts}_{rand}"
    return f"{ts}_{rand}"


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = "modifier_sim_explicit_genome_offspring_viability_per_site_env"
    run_tag = make_run_tag(args.run_tag)
    base = f"{stem}_{run_tag}"

    png = outdir / f"{base}.png"
    pdf = outdir / f"{base}.pdf"
    csv = outdir / f"{base}.csv"
    summary_csv = outdir / f"{base}_summary.csv"
    terminal_csv = outdir / f"{base}_terminal.csv"
    params_json = outdir / f"{base}_params.json"

    seed_base = args.seed_base
    if seed_base is None:
        seed_base = int(time.time_ns() % (2**31 - 1))

    tasks = []
    task_records = []
    for p_env_site in args.env_site_probs:
        F = 54 * (p_env_site ** 0.99)
        # F = (p_env_site/0.01639)*((p_env_site/0.015)**-.01)

        for init_U in args.starts:
            for r in range(args.reps):
                seed = seed_base + 97 * r + int(100 * init_U) + int(100000 * p_env_site)
                task = dict(
                    N=args.population,
                    n=args.genome_length,
                    G=args.generations,
                    init_U=init_U,
                    mu_floor=args.mu_floor,
                    mu_cap=args.mu_cap,
                    modifier_mut_prob=args.modifier_mut_prob,
                    log_step_sd=args.log_step_sd,
                    mismatch_cost=F,
                    p_env_site=p_env_site,
                    offspring_pool_factor=args.offspring_pool_factor,
                    seed=seed,
                )
                tasks.append(task)
                task_records.append({
                    "replicate": r,
                    "p_env_site": p_env_site,
                    "init_U": init_U,
                    "derived_mismatch_cost": F,
                    "seed": seed,
                })

    metadata = {
        "script_name": Path(__file__).name,
        "run_tag": run_tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_base": seed_base,
        "cli_args": vars(args),
        "derived_settings": {
            "filename_stem": stem,
            "base_output_name": base,
            "n_tasks": len(tasks),
        },
        "output_files": {
            "csv": str(csv),
            "summary_csv": str(summary_csv),
            "terminal_csv": str(terminal_csv),
            "png": str(png),
            "pdf": str(pdf),
            "params_json": str(params_json),
        },
        "task_records": task_records,
    }
    write_json(params_json, metadata)

    max_workers = max(1, args.workers)
    print(f"Launching {len(tasks)} simulations on {max_workers} workers")
    print(f"Run tag: {run_tag}")

    dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_one, t) for t in tasks]
        iterator = as_completed(futures)
        if args.no_progress:
            for f in iterator:
                dfs.append(f.result())
        else:
            for f in tqdm(iterator, total=len(futures), desc="Running simulation replicates"):
                dfs.append(f.result())

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(csv, index=False)

    summary = (
        df.groupby(["p_env_site", "mismatch_cost", "init_U", "generation"])
        .agg(mean_U=("mean_U", "mean"),
             sd_across_reps=("mean_U", "std"),
             mean_parent_mismatches=("mean_parent_mismatches", "mean"),
             mean_parent_fitness=("mean_parent_fitness", "mean"),
             expected_env_flips=("expected_env_flips", "mean"),
             n=("mean_U", "count"))
        .reset_index()
    )

    summary["se"] = summary["sd_across_reps"].fillna(0) / np.sqrt(summary["n"])
    summary["lo"] = summary["mean_U"] - 1.96 * summary["se"]
    summary["hi"] = summary["mean_U"] + 1.96 * summary["se"]
    summary.to_csv(summary_csv, index=False)

    terminal = (
        summary[summary["generation"] == summary["generation"].max()]
        .groupby(["p_env_site", "mismatch_cost"])
        .agg(mean_terminal_U=("mean_U", "mean"),
             mean_terminal_parent_mismatches=("mean_parent_mismatches", "mean"),
             mean_terminal_parent_fitness=("mean_parent_fitness", "mean"),
             expected_env_flips=("expected_env_flips", "mean"))
        .reset_index()
    )
    terminal.to_csv(terminal_csv, index=False)

    print("\nTerminal summary:")
    print(terminal.to_string(index=False))

    fig = plt.figure(figsize=(8.0, 6.8))

    ax1 = fig.add_subplot(2, 1, 1)
    ref_start = args.starts[1 if len(args.starts) > 1 else 0]
    for p_env_site in args.env_site_probs:
        if p_env_site == 0:
            label = "Static target"
        else:
            label = f"Changing target (E[flips/gen]={args.genome_length * p_env_site:.2f})"
        sub = summary[(summary["p_env_site"] == p_env_site) & (summary["init_U"] == ref_start)]
        ax1.plot(sub["generation"], sub["mean_parent_mismatches"], label=label)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Hamming distance to target")
    ax1.set_title("a.")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(2, 1, 2)
    for p_env_site in args.env_site_probs:
        label = "Static target" if p_env_site == 0 else "Changing target"
        for init_U in args.starts:
            sub = summary[(summary["p_env_site"] == p_env_site) & (summary["init_U"] == init_U)]
            ax2.plot(sub["generation"], sub["mean_U"], label=f"{label}, start={init_U:g}")
            ax2.fill_between(sub["generation"], sub["lo"], sub["hi"], alpha=0.12)

    ax2.axhline(args.genome_length * args.mu_floor, linestyle="--", linewidth=1)
    ax2.axhline(y=1, color="r", linestyle="--", linewidth=2, label="Threshold")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean genomic mutation intensity $U=n\\mu$")
    ax2.set_title("b.")
    ax2.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax2.set_xscale("log")

    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata["output_files_exist"] = {
        "csv": csv.exists(),
        "summary_csv": summary_csv.exists(),
        "terminal_csv": terminal_csv.exists(),
        "png": png.exists(),
        "pdf": pdf.exists(),
        "params_json": params_json.exists(),
    }
    metadata["terminal_summary_preview"] = terminal.to_dict(orient="records")
    write_json(params_json, metadata)

    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {terminal_csv}")
    print(f"Wrote {params_json}")


if __name__ == "__main__":
    main()
