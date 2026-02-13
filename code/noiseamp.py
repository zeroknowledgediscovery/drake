#!/usr/bin/env python3
import random, math, zlib
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

SEED = 2
ALPH = "ACGT"
n = 400
T = 300
mu = 0.01
reps = 30

# Selection: offspring must remain within this Hamming fraction of a structured template
MAX_MISMATCH_FRAC = 0.10   # 10% mismatches allowed
MAX_TRIES = 5000

random.seed(SEED)
np.random.seed(SEED)

template = ("ACGT" * (n // 4 + 1))[:n]

def mutate(seq, mu, alph=ALPH):
    s = list(seq)
    for i, ch in enumerate(s):
        if random.random() < mu:
            choices = [a for a in alph if a != ch]
            s[i] = random.choice(choices)
    return "".join(s)

def zlib_bits_per_symbol(seq):
    comp = zlib.compress(seq.encode("ascii"), level=9)
    return 8.0 * len(comp) / len(seq)

def bigram_cond_entropy_bits(seq):
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

def hamming_frac(a, b):
    return sum(1 for x, y in zip(a, b) if x != y) / len(a)

def passes_constraint(seq):
    return hamming_frac(seq, template) <= MAX_MISMATCH_FRAC

def simulate_lineage(mutation_only=True):
    x = template  # start structured
    Z, H, D = [], [], []
    for t in range(T + 1):
        Z.append(zlib_bits_per_symbol(x))
        H.append(bigram_cond_entropy_bits(x))
        D.append(hamming_frac(x, template))
        if t == T:
            break

        if mutation_only:
            x = mutate(x, mu)
        else:
            # viability filter: resample until constraint passes
            for _ in range(MAX_TRIES):
                y = mutate(x, mu)
                if passes_constraint(y):
                    x = y
                    break
            else:
                # fallback if constraint too strict: keep current
                x = x

    return np.array(Z), np.array(H), np.array(D)

Z_mut, H_mut, D_mut = [], [], []
Z_sel, H_sel, D_sel = [], [], []

for _ in range(reps):
    z, h, d = simulate_lineage(True)
    Z_mut.append(z); H_mut.append(h); D_mut.append(d)
    z, h, d = simulate_lineage(False)
    Z_sel.append(z); H_sel.append(h); D_sel.append(d)

Z_mut = np.vstack(Z_mut); H_mut = np.vstack(H_mut); D_mut = np.vstack(D_mut)
Z_sel = np.vstack(Z_sel); H_sel = np.vstack(H_sel); D_sel = np.vstack(D_sel)

def mean_ci(arr):
    m = arr.mean(axis=0)
    s = arr.std(axis=0, ddof=1)
    se = s / math.sqrt(arr.shape[0])
    z = 1.96
    return m, m - z * se, m + z * se

x = np.arange(T + 1)
mZ0, loZ0, hiZ0 = mean_ci(Z_mut)
mH0, loH0, hiH0 = mean_ci(H_mut)
mD0, loD0, hiD0 = mean_ci(D_mut)

mZ1, loZ1, hiZ1 = mean_ci(Z_sel)
mH1, loH1, hiH1 = mean_ci(H_sel)
mD1, loD1, hiD1 = mean_ci(D_sel)

plt.figure(figsize=(10.5, 3.8))

plt.subplot(1, 3, 1)
plt.plot(x, mZ0, label="Mutation only")
plt.fill_between(x, loZ0, hiZ0, alpha=0.2)
plt.plot(x, mZ1, label="Mutation + viability filter")
plt.fill_between(x, loZ1, hiZ1, alpha=0.2)
plt.xlabel("Generation")
plt.ylabel("Compressed bits/symbol (zlib)")
plt.title("Compressibility proxy")
plt.legend(frameon=False)

plt.subplot(1, 3, 2)
plt.plot(x, mH0, label="Mutation only")
plt.fill_between(x, loH0, hiH0, alpha=0.2)
plt.plot(x, mH1, label="Mutation + viability filter")
plt.fill_between(x, loH1, hiH1, alpha=0.2)
plt.xlabel("Generation")
plt.ylabel(r"$H(X_t\mid X_{t-1})$ (bits/symbol)")
plt.title("Entropy proxy")
plt.legend(frameon=False)

plt.subplot(1, 3, 3)
plt.plot(x, mD0, label="Mutation only")
plt.fill_between(x, loD0, hiD0, alpha=0.2)
plt.plot(x, mD1, label="Mutation + viability filter")
plt.fill_between(x, loD1, hiD1, alpha=0.2)
plt.xlabel("Generation")
plt.ylabel("Hamming distance to template")
plt.title("Distance from structure")
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("panel_option1_template_filter.png", dpi=300)
plt.show()

print("Saved: panel_option1_template_filter.png")
print(f"Params: n={n}, T={T}, mu={mu}, MAX_MISMATCH_FRAC={MAX_MISMATCH_FRAC}, reps={reps}")
