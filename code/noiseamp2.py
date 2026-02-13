#!/usr/bin/env python3
"""
One-panel Option 1 (toy simulation) for SI Fig. S1

Goal: illustrate drift toward a high-entropy regime under mutation alone,
and boundedness under a simple viability constraint, without making any
empirical claim about real genomes.

Panel: empirical bigram conditional entropy H(X_t | X_{t-1}) (bits/symbol)
over generations, for:
  (A) mutation only
  (B) mutation + viability filter (accept offspring only if close to a structured template)

Outputs:
  - figS1_toy_entropy_proxy.png
  - figS1_toy_entropy_proxy.pdf
"""

import random
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (tune as needed)
# ----------------------------
SEED = 2
ALPH = "ACGT"
n = 400                 # sequence length
G = 300                 # generations
mu = 1.0 / n            # try 1/n, 2/n, 5/n to show robustness
reps = 50               # replicate lineages

# Viability constraint: stay within a Hamming band around a structured template
MAX_MISMATCH_FRAC = 0.10   # 10% mismatches allowed
MAX_TRIES = 20000          # resample attempts per generation if constraint fails

# Confidence interval across replicates (normal approx)
Z_95 = 1.96

random.seed(SEED)
np.random.seed(SEED)

# Structured template: periodic sequence
template = ("ACGT" * (n // 4 + 1))[:n]


# ----------------------------
# Helpers
# ----------------------------
def mutate(seq: str, mu_: float, alph: str = ALPH) -> str:
    s = list(seq)
    for i, ch in enumerate(s):
        if random.random() < mu_:
            # mutate to a different symbol uniformly
            choices = [a for a in alph if a != ch]
            s[i] = random.choice(choices)
    return "".join(s)

def hamming_frac(a: str, b: str) -> float:
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)

def passes_constraint(seq: str) -> bool:
    return hamming_frac(seq, template) <= MAX_MISMATCH_FRAC

def bigram_cond_entropy_bits(seq: str) -> float:
    """
    Empirical H(X_t | X_{t-1}) in bits per symbol using adjacent bigrams.
    """
    if len(seq) < 2:
        return 0.0
    pair_counts = Counter(zip(seq[:-1], seq[1:]))
    prev_counts = Counter(seq[:-1])
    total = len(seq) - 1

    H = 0.0
    for (a, b), c_ab in pair_counts.items():
        p_ab = c_ab / total
        p_b_given_a = c_ab / prev_counts[a]
        H -= p_ab * math.log2(p_b_given_a)
    return H

def simulate_lineage(mutation_only: bool) -> np.ndarray:
    x = template  # start from structured state
    H = np.empty(G + 1, dtype=float)

    for t in range(G + 1):
        H[t] = bigram_cond_entropy_bits(x)
        if t == G:
            break

        if mutation_only:
            x = mutate(x, mu)
        else:
            # viability filter: resample until constraint passes
            accepted = False
            for _ in range(MAX_TRIES):
                y = mutate(x, mu)
                if passes_constraint(y):
                    x = y
                    accepted = True
                    break
            if not accepted:
                # if constraint is too strict for chosen mu, keep current
                x = x

    return H

def mean_ci(arr: np.ndarray, z: float = Z_95):
    m = arr.mean(axis=0)
    s = arr.std(axis=0, ddof=1)
    se = s / math.sqrt(arr.shape[0])
    return m, m - z * se, m + z * se


# ----------------------------
# Run replicates
# ----------------------------
H_mut = []
H_sel = []

for _ in range(reps):
    H_mut.append(simulate_lineage(mutation_only=True))
    H_sel.append(simulate_lineage(mutation_only=False))

H_mut = np.vstack(H_mut)
H_sel = np.vstack(H_sel)

m0, lo0, hi0 = mean_ci(H_mut)
m1, lo1, hi1 = mean_ci(H_sel)

# ----------------------------
# Plot (single panel)
# ----------------------------
x = np.arange(G + 1)

plt.figure(figsize=(5.2, 3.2))
plt.plot(x, m0, label="Mutation only")
plt.fill_between(x, lo0, hi0, alpha=0.2)

plt.plot(x, m1, label="Mutation + viability filter")
plt.fill_between(x, lo1, hi1, alpha=0.2)

plt.xlabel("Generation")
plt.ylabel(r"Bigram conditional entropy $H(X_t\mid X_{t-1})$ (bits/symbol)")
plt.title("Toy illustration: drift vs retention")
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("figS1_toy_entropy_proxy.png", dpi=300)
plt.savefig("figS1_toy_entropy_proxy.pdf")
plt.show()

print("Saved: figS1_toy_entropy_proxy.png and figS1_toy_entropy_proxy.pdf")
print(f"Params: n={n}, G={G}, mu={mu:.6g}, reps={reps}, MAX_MISMATCH_FRAC={MAX_MISMATCH_FRAC}")
