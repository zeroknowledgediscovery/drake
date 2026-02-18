import numpy as np
import matplotlib.pyplot as plt
import math

def log_binom(n, k):
    return math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)

def log_Hm(n, m, q):
    # log |H_m| = log C(n,m) + m log(q-1)
    return log_binom(n, m) + m*math.log(q-1)

def phi_theory(mu, n, q, Delta, C=1.0):
    # Exact expression used in the draft:
    # Phi ~ C 2^{-Delta} (1-mu)^n sum_{m=1}^n (mu/(q-1))^m
    r = mu/(q-1)
    if r >= 1:
        return np.nan
    s = r*(1 - r**n)/(1 - r)   # geometric sum to n
    return C*(2**(-Delta))*((1-mu)**n)*s

def phi_mc(mu, n, q, Delta, rng, trials=100000, kappa=1.0):
    # Monte Carlo proxy:
    # M ~ Bin(n,mu)
    # discovery given M=m is Bernoulli(p_m), with p_m = kappa * 2^{-Delta} / |H_m|
    M = rng.binomial(n, mu, size=trials)
    uniq, counts = np.unique(M, return_counts=True)
    disc = 0
    for m, c in zip(uniq, counts):
        if m == 0:
            continue
        p = kappa*(2**(-Delta))*math.exp(-log_Hm(n, int(m), q))
        if p > 1.0:
            p = 1.0
        if p <= 0.0:
            continue
        disc += rng.binomial(int(c), p)
    return disc / trials



def phi_mc(mu, n, q, Delta, rng, trials=100000, kappa=1.0):
    M = rng.binomial(n, mu, size=trials)
    uniq, counts = np.unique(M, return_counts=True)

    disc_exp = 0.0
    for m, c in zip(uniq, counts):
        if m == 0:
            continue
        m = int(m)
        logHm = log_Hm(n, m, q)
        p = kappa*(2**(-Delta))*math.exp(-logHm)  # = kappa*2^-Delta/|H_m|
        if p <= 0.0:
            continue
        if p > 1.0:
            p = 1.0
        disc_exp += c * p

    return disc_exp / trials


# -----------------------
# Parameters
# -----------------------
q = 4
Delta = 10
ns = [10**3, 10**4, 10**5]

# "temperature" grid and corresponding mu = T/n
T_grid = np.logspace(-2, 2, 41)          # 0.01 .. 100
mus = {n: np.clip(T_grid/n, 1e-15, 0.49) for n in ns}

rng = np.random.default_rng(7)

# -----------------------
# Compute theory + MC
# -----------------------
phi_mc_vals = {}
phi_th_vals = {}
for n in ns:
    phi_mc_vals[n] = np.array([
        phi_mc(float(mu), n, q, Delta, rng, trials=100000, kappa=1.0)
        for mu in mus[n]
    ])
    phi_th_vals[n] = np.array([
        phi_theory(float(mu), n, q, Delta, C=1.0)
        for mu in mus[n]
    ])

# -----------------------
# Plot: Φ vs μ and Φ vs T
# -----------------------
L=.9
fig = plt.figure(figsize=(L*4,L*7))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

for n in ns:
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.plot(mus[n], phi_th_vals[n], linestyle="-", label=f"n={n}")
    #ax1.scatter(mus[n], phi_mc_vals[n], s=14, label=f"MC n={n}")

ax1.set_xlabel("Per-site mutation rate μ")
ax1.set_ylabel("Discovery rate Φ(μ)")
ax1.set_title("a.")
ax1.legend(fontsize=10, loc="best")

for n in ns:
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.plot(T_grid, phi_th_vals[n], linestyle="-", label=f"n={n}")
    #ax2.scatter(T_grid, phi_mc_vals[n], s=14, label=f"MC n={n}")

ax2.set_xlabel("Mutation temperature T = nμ")
ax2.set_ylabel("Discovery rate Φ")
ax2.set_title("b.")

# Add Te^{-T} guide curve (scaled to match peak)
T = T_grid
guide = T * np.exp(-T)
scale = np.nanmax(phi_th_vals[10**4]) / np.max(guide)
ax2.plot(T, scale * guide, '--k',label='$Te^{-T}$' )

ax2.legend(fontsize=10, loc="best")
fig.tight_layout()



import pandas as pd

# -----------------------
# Assemble all data into one DataFrame
# -----------------------
rows = []

for n in ns:
    for i, T in enumerate(T_grid):
        mu = float(mus[n][i])
        phi_th = float(phi_th_vals[n][i])
        phi_mc_v = float(phi_mc_vals[n][i])
        
        rows.append({
            "n": n,
            "T": T,
            "mu": mu,
            "phi_theory": phi_th,
            "phi_mc": phi_mc_v
        })

df_out = pd.DataFrame(rows)

# Optional: sort cleanly
df_out = df_out.sort_values(["n", "T"]).reset_index(drop=True)

# -----------------------
# Save to CSV
# -----------------------
df_out.to_csv("temperature_scaling_sim_data.csv", index=False)

print("Wrote temperature_scaling_sim_data.csv")









fig.savefig("temperature_scaling_sim.png", dpi=300, bbox_inches='tight',transparent=True)
fig.savefig("../pnas/Figures/temperature_scaling_sim.pdf", bbox_inches='tight',transparent=True)
print("Wrote temperature_scaling_sim.png and temperature_scaling_sim.pdf")
