"""
ks_analysis.py  (bootstrapped convergence edition)
====================================================
Determines the minimum number of evaluation episodes required for the
reward mean and IQM to stabilise with high confidence, using Bootstrapped
Confidence Intervals.

RELATIONSHIP TO AGARWAL ET AL. (2021)
--------------------------------------
Agarwal et al. (2021) "Deep Reinforcement Learning at the Edge of the
Statistical Precipice", NeurIPS 2021.  https://arxiv.org/abs/2108.13264

What the paper DOES propose (their actual method):
  - Comparing MULTIPLE ALGORITHMS across MULTIPLE TASKS (e.g. 57 Atari games)
  - "Stratified bootstrap" = resampling *tasks* with replacement, then
    resampling *training seeds* within each resampled task
  - Recommended aggregate metric: IQM (Interquartile Mean), NOT the mean
    (mean is their LEAST recommended metric — highly sensitive to outliers)
  - Their sample-size analysis (Figure 3) answers "how many training seeds?"
    not "how many evaluation episodes?"

What this script borrows from the paper:
  - The percentile bootstrap CI machinery (same principle, applied per-episode)
  - The recommendation to use IQM instead of the mean
  - B ≥ 2000 resamples for stable CIs

What this script does differently:
  - We bootstrap over EPISODES within one model evaluation (not seeds/tasks)
  - The bootstrap is plain (not stratified), because we have one task/one model
  - We use IQM and mean both, for completeness

SUFFICIENCY THRESHOLD
---------------------
The paper offers no specific threshold for this use case, so we use:

  CI width < SUFFICIENCY_SD_FRACTION × std(rewards)

  e.g. SUFFICIENCY_SD_FRACTION = 0.3 means the 95% CI of the IQM must span
  less than 30% of one standard deviation — a scale-aware, outlier-robust
  criterion.  Tighten (smaller value) for more precision, loosen for fewer
  required episodes.

  A range-based threshold (e.g. 5% of max−min) is tempting but problematic:
  a single outlier episode inflates max−min and makes the threshold easier
  to pass artificially.

WORKFLOW
--------
1. Run evaluate.py (NUM_EVAL_EPISODES=600).  It saves episode_rewards.npy.
2. Run this script.

The 600 episodes are split ONCE:
  Probe pool  : episodes   1-300  (swept at increasing sample sizes)
  Reference   : episodes 301-600  (held-out for descriptive comparison only)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# ▶ USER-TUNEABLE PARAMETERS
# ---------------------------------------------------------------------------
B = 2000        # Bootstrap resamples (Agarwal et al. use ≥2000)
STEP = 5        # Episode increment for the convergence sweep
ALPHA = 0.05    # Significance level → 95% CI

# Sufficiency threshold: CI width of the IQM must be less than this
# fraction of one standard deviation of all rewards.
# 0.3 = "CI spans < 30% of 1σ" — a scale-aware, outlier-robust criterion.
# Tighten to 0.2 for more precision; loosen to 0.5 for fewer episodes needed.
SUFFICIENCY_SD_FRACTION = 0.3   # ← toggle this

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir   = os.path.dirname(os.path.abspath(__file__))
rewards_path = os.path.join(script_dir, "episode_rewards.npy")
output_path  = os.path.join(script_dir, "bootstrap_convergence.png")

if not os.path.exists(rewards_path):
    raise FileNotFoundError(
        f"episode_rewards.npy not found at:\n  {rewards_path}\n"
        "Please run evaluate.py (NUM_EVAL_EPISODES=600) first."
    )

rewards = np.load(rewards_path)
N = len(rewards)
if N < 20:
    raise ValueError(f"Need at least 20 episodes, got {N}.")

print(f"Loaded {N} episode rewards from:\n  {rewards_path}\n")

# ---------------------------------------------------------------------------
# Split into independent halves
# (Agarwal et al. emphasise that evaluation samples must be independent
#  to produce unbiased interval estimates.)
# ---------------------------------------------------------------------------
half       = N // 2
probe_pool = rewards[:half]   # episodes 1–100  → sub-sampled in the sweep
reference  = rewards[half:]   # episodes 101–200 → held-out descriptive set

print(f"Probe pool : episodes 1–{half}       (n={len(probe_pool)})")
print(f"Reference  : episodes {half+1}–{N}  (n={len(reference)})\n")

# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
reward_range          = rewards.max() - rewards.min()
reward_std            = rewards.std()
sufficiency_threshold = SUFFICIENCY_SD_FRACTION * reward_std

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
for label, arr in [
    (f"All {N} episodes",            rewards),
    (f"Probe pool  (ep 1-{half})",   probe_pool),
    (f"Reference   (ep {half+1}-{N})", reference),
]:
    print(f"  {label}:")
    print(f"    N={len(arr):3d}  mean={np.mean(arr):.3f}  IQM={np.mean(arr[int(len(arr)*0.25):int(len(arr)*0.75)+1]):.3f}  "
          f"std={np.std(arr):.3f}  "
          f"[{np.min(arr):.3f}, {np.max(arr):.3f}]")

print(f"\n  Reward std (all episodes) : {reward_std:.3f}")
print(f"  Reward range (max-min)    : {reward_range:.3f}  "
      f"  ← NOTE: range is outlier-sensitive, not used as threshold")
print(f"  Sufficiency threshold     : {sufficiency_threshold:.3f}  "
      f"({SUFFICIENCY_SD_FRACTION} × σ  = CI width < {SUFFICIENCY_SD_FRACTION*100:.0f}% of 1 std dev)")
print()

# ---------------------------------------------------------------------------
# ── BOOTSTRAP ENGINE ──────────────────────────────────────────────────────
# We bootstrap the IQM (Interquartile Mean) as recommended by Agarwal et al.
# (2021), who show IQM is more robust and efficient than the mean for RL
# evaluation.  IQM = mean of the middle 50% of scores (discards bottom and
# top 25%), making it resistant to extreme outlier episodes.
#
# Plain percentile bootstrap (not stratified) because we have a single task
# and single model — the stratification in the paper applies across tasks.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)   # reproducible results

def iqm(arr: np.ndarray) -> float:
    """Interquartile Mean: mean of the middle 50% of values.
    Agarwal et al. (2021) recommend this over mean for RL evaluation
    because it is robust to outlier episodes.
    """
    lo, hi = np.percentile(arr, 25), np.percentile(arr, 75)
    middle = arr[(arr >= lo) & (arr <= hi)]
    return middle.mean() if len(middle) > 0 else arr.mean()

def bootstrap_ci(sample: np.ndarray, B: int, alpha: float):
    """
    Compute bootstrapped IQM (and mean) with (1-alpha)*100% CI.

    Returns
    -------
    mean_val  : float – sample mean
    iqm_val   : float – sample IQM
    iqm_ci_lo : float – lower CI bound of IQM
    iqm_ci_hi : float – upper CI bound of IQM
    iqm_width : float – CI width of IQM
    mean_width: float – CI width of mean (for comparison)
    """
    boot_iqms  = np.empty(B)
    boot_means = np.empty(B)
    for i in range(B):
        resample     = rng.choice(sample, size=len(sample), replace=True)
        boot_iqms[i] = iqm(resample)
        boot_means[i]= resample.mean()

    iqm_lo = np.percentile(boot_iqms,  100 * alpha / 2)
    iqm_hi = np.percentile(boot_iqms,  100 * (1 - alpha / 2))
    m_lo   = np.percentile(boot_means, 100 * alpha / 2)
    m_hi   = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return (sample.mean(), iqm(sample),
            iqm_lo, iqm_hi, iqm_hi - iqm_lo,
            m_hi - m_lo)

# ---------------------------------------------------------------------------
# Convergence sweep
# ---------------------------------------------------------------------------
sample_sizes = list(range(STEP, len(probe_pool) + 1, STEP))

means      = []
iqms       = []
iqm_ci_los = []
iqm_ci_his = []
iqm_widths = []
mean_widths= []

print(f"Running bootstrap sweep  (B={B}, {int((1-ALPHA)*100)}% CI, "
      f"sufficiency = IQM CI width < {sufficiency_threshold:.3f}) …")

for n in sample_sizes:
    subset = probe_pool[:n]
    m, iq, iq_lo, iq_hi, iq_w, m_w = bootstrap_ci(subset, B, ALPHA)
    means.append(m)
    iqms.append(iq)
    iqm_ci_los.append(iq_lo)
    iqm_ci_his.append(iq_hi)
    iqm_widths.append(iq_w)
    mean_widths.append(m_w)

# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print(f"BOOTSTRAP CONVERGENCE  (Agarwal et al. 2021: use IQM, not mean)")
print(f"  Sufficient when IQM CI width < {SUFFICIENCY_SD_FRACTION} × σ = {sufficiency_threshold:.3f}")
print("=" * 90)
print(f"{'N':>6}  {'Mean':>8}  {'IQM':>8}  {'IQM CI lo':>10}  {'IQM CI hi':>10}  "
      f"{'IQM CI w':>10}  {'Sufficient?':>12}")
print("-" * 90)

first_sufficient = None
for i, n in enumerate(sample_sizes):
    ok     = iqm_widths[i] < sufficiency_threshold
    marker = "✅ yes" if ok else "❌ no "
    if ok and first_sufficient is None:
        first_sufficient = n
    print(f"{n:>6}  {means[i]:>8.3f}  {iqms[i]:>8.3f}  {iqm_ci_los[i]:>10.3f}  "
          f"{iqm_ci_his[i]:>10.3f}  {iqm_widths[i]:>10.3f}  {marker:>12}")

print("=" * 90)

if first_sufficient is not None:
    print(f"\n📌 First sufficient N : {first_sufficient} episodes  "
          f"(IQM 95% CI width < {SUFFICIENCY_SD_FRACTION}σ = {sufficiency_threshold:.3f})")
else:
    print(f"\n📌 No sample size reached sufficiency — "
          f"try loosening SUFFICIENCY_SD_FRACTION or collecting more episodes.")

_, iq_full, iq_lo_f, iq_hi_f, iq_w_f, m_w_f = bootstrap_ci(probe_pool, B, ALPHA)
print(f"\nFull probe pool ({half} ep):")
print(f"  Mean = {probe_pool.mean():.3f},  IQM = {iq_full:.3f}")
print(f"  IQM 95% CI = [{iq_lo_f:.3f}, {iq_hi_f:.3f}],  width = {iq_w_f:.3f}  "
      f"({'✅ sufficient' if iq_w_f < sufficiency_threshold else '❌ not sufficient'})")
print(f"  Mean 95% CI width = {m_w_f:.3f}  "
      f"(wider than IQM CI: {m_w_f:.3f} vs {iq_w_f:.3f} — IQM is more efficient)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
x = np.array(sample_sizes)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.subplots_adjust(hspace=0.55, top=0.93)

fig.suptitle(
    "Bootstrap Confidence Interval Convergence Analysis",
    fontsize=12, fontweight="bold"
)

# ── Plot 1: IQM + shaded 95% CI only (mean removed — too cluttered) ───────
ax1.fill_between(x, iqm_ci_los, iqm_ci_his,
                 alpha=0.20, color="steelblue")
ax1.plot(x, iqms, color="steelblue", linewidth=2, label="IQM (bootstrapped)")
ax1.axhline(iq_full, color="navy", linestyle="--", linewidth=1.0,
            label=f"Probe-pool IQM = {iq_full:.3f}")

ax1.set_ylabel("IQM Episode Reward", fontsize=10)
ax1.set_title("IQM with 95% Bootstrapped Confidence Interval", fontsize=10)
ax1.legend(fontsize=9, loc="lower right")
ax1.grid(True, alpha=0.25)

# ── Plot 2: CI Width — IQM vs Mean, clean separation ─────────────────────
ax2.plot(x, iqm_widths,  color="steelblue", linewidth=2,
         label="IQM 95% CI width")
ax2.plot(x, mean_widths, color="grey",      linewidth=1.5, linestyle="--",
         label="Mean 95% CI width")

ax2.set_xlabel("Number of Evaluation Episodes", fontsize=10)
ax2.set_ylabel("95% CI Width", fontsize=10)
ax2.set_title("Uncertainty Reduction — IQM CI Width vs. Sample Size", fontsize=10)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.25)
ax2.set_ylim(bottom=0)

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\n📊 Plot saved to: {output_path}")
plt.show()

