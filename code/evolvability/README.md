# Mutation--Modifier Simulation Framework

## Overview

This repository implements a stochastic population-genetic simulation
designed to study the evolution of mutation rates under selection,
environmental change, and modifier dynamics.

The model explicitly represents genomes, mutation processes, and
viability selection, allowing mutation rates themselves to evolve over
time through modifier mutations.

The primary scientific objective is to examine whether populations
converge toward a characteristic genomic mutation intensity:

U = nμ

and how that equilibrium depends on environmental volatility and
selection strength.

------------------------------------------------------------------------

## Quick Start

Run a typical experiment:

python3 simulate_rate_evolution.py\
--env-site-probs 0.01 0.015 0.02\
--population 5000\
--genome-length 80\
--generations 10000\
--reps 64\
--workers 8\
--outdir results

------------------------------------------------------------------------

## Core Simulation Logic

At each generation:

1)  Fitness is computed from mismatch distance to a target sequence.
2)  Parents are selected probabilistically based on fitness.
3)  Offspring genomes mutate independently at each site.
4)  Mutation rates themselves evolve through modifier mutations.
5)  Offspring undergo viability selection.
6)  The environment may change via random target-site flips.

Fitness function:

fitness = exp( − mismatch_cost × mismatches )

Expected environmental change:

expected_flips = genome_length × p_env_site

------------------------------------------------------------------------

## Command Reference

General form:

python3 simulate_rate_evolution.py \[OPTIONS\]

### Core Parameters

--population\
Population size

--genome-length\
Genome length

--generations\
Number of generations simulated

--reps\
Number of replicates

--workers\
Number of parallel processes

### Mutation Parameters

--starts\
Initial genomic mutation intensities

--mu-floor\
Minimum allowed mutation rate

--mu-cap\
Maximum allowed mutation rate

--modifier-mut-prob\
Probability of modifier mutation

--log-step-sd\
Standard deviation of modifier step size

### Selection Parameters

--mismatch-cost\
Fitness penalty per mismatch

--offspring-pool-factor\
Number of candidate offspring per survivor

### Environmental Parameters

--env-site-probs\
Probability of environmental change per site

------------------------------------------------------------------------

## Output Files

Each run produces:

raw trajectory CSV\
summary CSV\
terminal summary CSV\
PNG figure\
PDF figure\
parameter JSON metadata

------------------------------------------------------------------------

## Typical Workflow

1)  Run simulation

python3 simulate_rate_evolution.py ...

2)  Recompute summary

python3 recompute_summary_log.py raw.csv

3)  Generate plot

python3 plot_panel_b.py summary.csv

------------------------------------------------------------------------

## Reproducibility

Every run writes a parameters JSON file containing:

-   command-line arguments
-   derived parameters
-   random seeds
-   output filenames
-   timestamp

This ensures exact reproducibility of simulation outputs.
