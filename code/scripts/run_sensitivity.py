"""
run_sensitivity.py — Conley-Hansen-Rossi (2012) Exclusion Restriction
                     Sensitivity Analysis

Addresses the known weakness of the RDD/IV analysis: the October 2013 cutoff
changed both the drug (ranibizumab -> aflibercept) AND the dosing protocol
(PRN -> treat-and-extend), violating the exclusion restriction.

The CHR framework relaxes the standard IV exclusion restriction by allowing
the instrument (era) to have a direct effect on the outcome (gamma):

    Standard:   Y = Xb + Dt + e,         E[Ze] = 0
    CHR:        Y = Xb + Dt + Zg + e,    E[Ze] = 0

The adjusted LATE becomes:  LATE(g) = (ITT - g) / first_stage

Key innovation: ranibizumab patients exist in BOTH eras (3,261 pre + 590 post),
so gamma is directly estimable from the data — among ranibizumab-only patients,
the entire era effect is gamma (since the drug is held constant).

Reference:
    Conley, T.G., Hansen, C.B. & Rossi, P.E. (2012). Plausibly exogenous.
    Review of Economics and Statistics, 94(1), 260-272.

Outputs:
    output/tables/sensitivity_placebo_test.csv
    output/tables/sensitivity_gamma_estimates.csv
    output/tables/sensitivity_chr_results.csv
    output/tables/sensitivity_uci.csv
    output/tables/sensitivity_summary.csv
    output/figures/chr_sensitivity.png
    output/figures/era_placebo.png
    output/figures/multi_outcome_sensitivity.png
    output/logs/log_sensitivity_<timestamp>.log

Usage:
    python3 code/scripts/run_sensitivity.py
"""

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PATIENT_CSV = ROOT / "output" / "results" / "cohort_patient.csv"
VISITS_CSV = ROOT / "output" / "results" / "cohort_visits.csv.gz"
OUT_RESULTS = ROOT / "output" / "results"
OUT_TABLES = ROOT / "output" / "tables"
OUT_FIGURES = ROOT / "output" / "figures"
OUT_LOGS = ROOT / "output" / "logs"

for d in [OUT_RESULTS, OUT_TABLES, OUT_FIGURES, OUT_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT_LOGS / f"log_sensitivity_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


# ===================================================================
# Helper: manual 2SLS (copied from run_rdd.py)
# ===================================================================
def iv_2sls(y, d, z, x=None):
    """Two-stage least squares with robust standard errors.

    Parameters
    ----------
    y : array — outcome
    d : array — endogenous treatment
    z : array — instrument(s) (excluded)
    x : array or None — exogenous controls (included in both stages)

    Returns
    -------
    dict with keys: coef, se, t, p, ci_lo, ci_hi, f_first, n
    """
    n = len(y)
    y, d, z = np.asarray(y, float), np.asarray(d, float), np.asarray(z, float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Build exogenous regressors (constant + controls)
    if x is not None:
        x = np.asarray(x, float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        W = np.column_stack([np.ones(n), x])
    else:
        W = np.ones((n, 1))

    # --- First stage: D = W*gamma + Z*delta + u ---
    first_X = np.column_stack([W, z])
    first_model = sm.OLS(d, first_X).fit(cov_type="HC1")
    d_hat = first_model.fittedvalues

    # F-statistic on excluded instruments
    n_excl = z.shape[1]
    r_matrix = np.zeros((n_excl, first_X.shape[1]))
    for i in range(n_excl):
        r_matrix[i, W.shape[1] + i] = 1
    f_test = first_model.f_test(r_matrix)
    f_first = float(f_test.fvalue)

    # --- Second stage: Y = W*beta + D_hat*tau + e ---
    second_X = np.column_stack([W, d_hat])
    second_ols = sm.OLS(y, second_X).fit()

    # Correct SEs: use D not D_hat for residuals
    actual_X = np.column_stack([W, d])
    resid_corrected = y - actual_X @ second_ols.params

    # Robust variance using corrected residuals
    bread = np.linalg.inv(second_X.T @ second_X)
    meat = np.zeros((second_X.shape[1], second_X.shape[1]))
    for i in range(n):
        xi = second_X[i:i + 1].T
        meat += (resid_corrected[i] ** 2) * (xi @ xi.T)
    # HC1 correction
    meat *= n / (n - second_X.shape[1])
    V = bread @ meat @ bread

    tau_idx = -1  # treatment coefficient is last
    coef = second_ols.params[tau_idx]
    se = np.sqrt(V[tau_idx, tau_idx])
    t_stat = coef / se
    p_val = 2 * stats.t.sf(abs(t_stat), df=n - second_X.shape[1])
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    return {
        "coef": coef, "se": se, "t": t_stat, "p": p_val,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "f_first": f_first, "n": n,
        "first_stage_model": first_model,
        "second_stage_model": second_ols,
    }


# ===================================================================
# 1. Load data + construct injection intensity features
# ===================================================================
log.info("=" * 60)
log.info("LOADING DATA")
log.info("=" * 60)

pt = pd.read_csv(PATIENT_CSV)
log.info(f"Loaded {len(pt)} patients")

visits = pd.read_csv(VISITS_CSV)
log.info(f"Loaded {len(visits)} visits")

# --- Injection intensity features from visit-level data ---
log.info("Computing injection intensity features from visit data...")

# Count injections within first 2 years (730 days)
visits_2yr = visits[visits["time"] <= 730]
inj_2yr = visits_2yr.groupby("anon_id").agg(
    n_injections_2yr=("injection_given", "sum"),
    n_visits_2yr=("time", "count"),
).reset_index()

pt = pt.merge(inj_2yr, on="anon_id", how="left")
pt["n_injections_2yr"] = pt["n_injections_2yr"].fillna(0)
pt["n_visits_2yr"] = pt["n_visits_2yr"].fillna(0)

# Injection rate (per year of follow-up)
pt["injection_rate"] = np.where(
    pt["follow_up_years"] > 0,
    pt["n_injections"] / pt["follow_up_years"],
    0,
)

# Monitoring ratio: fraction of visits without injection
pt["monitoring_ratio"] = np.where(
    pt["n_visits"] > 0,
    1 - (pt["n_injections"] / pt["n_visits"]),
    0,
)

log.info(f"  n_injections_2yr: mean={pt['n_injections_2yr'].mean():.1f}, "
         f"sd={pt['n_injections_2yr'].std():.1f}")
log.info(f"  injection_rate:   mean={pt['injection_rate'].mean():.2f}/yr, "
         f"sd={pt['injection_rate'].std():.2f}")
log.info(f"  monitoring_ratio: mean={pt['monitoring_ratio'].mean():.3f}, "
         f"sd={pt['monitoring_ratio'].std():.3f}")

# Key variables
covariates_baseline = [
    "va_baseline", "female", "age_midpoint", "loaded_binary",
    "induction_interval", "eth_afro_caribbean", "eth_south_east_asian",
    "eth_unknown", "eth_other",
]
covariates_intensity = ["n_injections_2yr", "injection_rate", "monitoring_ratio"]

Z = pt["era_post2013"].values
D = pt["treatment"].values

# Summary by era and treatment
log.info(f"\n  Era x Treatment crosstab:")
log.info(f"  Pre-2013  ranibizumab: {((Z == 0) & (D == 0)).sum()}")
log.info(f"  Pre-2013  aflibercept: {((Z == 0) & (D == 1)).sum()}")
log.info(f"  Post-2013 ranibizumab: {((Z == 1) & (D == 0)).sum()}")
log.info(f"  Post-2013 aflibercept: {((Z == 1) & (D == 1)).sum()}")


# ===================================================================
# 2. Era placebo test (ranibizumab-only)
# ===================================================================
log.info("\n" + "=" * 60)
log.info("ERA PLACEBO TEST (ranibizumab-only patients)")
log.info("=" * 60)
log.info("If era affects outcomes among ranibizumab patients, the")
log.info("exclusion restriction is violated (era has a direct effect).")

rani = pt[pt["treatment"] == 0].copy()
log.info(f"Ranibizumab patients: {len(rani)} "
         f"(pre-2013: {(rani['era_post2013'] == 0).sum()}, "
         f"post-2013: {(rani['era_post2013'] == 1).sum()})")

outcomes_placebo = {
    "va_change": {"data": rani, "label": "VA change (letters)"},
}
# Add binary outcomes for eligible subsets
rani_70 = rani[rani["cohort_eligible_va70"] == 1].dropna(subset=["event_va70"])
if len(rani_70) > 0 and rani_70["era_post2013"].nunique() == 2:
    outcomes_placebo["event_va70"] = {
        "data": rani_70, "label": "Achieved VA >= 70",
    }
rani_35 = rani[rani["cohort_eligible_va35"] == 1].dropna(subset=["event_va35"])
if len(rani_35) > 0 and rani_35["era_post2013"].nunique() == 2:
    outcomes_placebo["event_va35"] = {
        "data": rani_35, "label": "Deteriorated to VA <= 35",
    }

placebo_rows = []

for outcome_col, info in outcomes_placebo.items():
    df_sub = info["data"]
    y_sub = df_sub[outcome_col].values
    z_sub = df_sub["era_post2013"].values

    log.info(f"\n  Outcome: {info['label']} (n={len(df_sub)})")

    # Spec 1: unadjusted
    X1 = sm.add_constant(z_sub)
    m1 = sm.OLS(y_sub, X1).fit(cov_type="HC1")
    log.info(f"    Unadjusted:         era coef = {m1.params[1]:+.4f} "
             f"(SE={m1.bse[1]:.4f}, p={m1.pvalues[1]:.4f})")
    placebo_rows.append({
        "outcome": outcome_col, "specification": "unadjusted",
        "era_coef": m1.params[1], "era_se": m1.bse[1],
        "era_pval": m1.pvalues[1], "n": len(df_sub),
    })

    # Spec 2: baseline covariates
    X_base = df_sub[covariates_baseline].values
    X2 = np.column_stack([np.ones(len(z_sub)), X_base, z_sub])
    m2 = sm.OLS(y_sub, X2).fit(cov_type="HC1")
    log.info(f"    + baseline covs:    era coef = {m2.params[-1]:+.4f} "
             f"(SE={m2.bse[-1]:.4f}, p={m2.pvalues[-1]:.4f})")
    placebo_rows.append({
        "outcome": outcome_col, "specification": "baseline_covariates",
        "era_coef": m2.params[-1], "era_se": m2.bse[-1],
        "era_pval": m2.pvalues[-1], "n": len(df_sub),
    })

    # Spec 3: baseline + intensity covariates
    X_int = df_sub[covariates_intensity].values
    X3 = np.column_stack([np.ones(len(z_sub)), X_base, X_int, z_sub])
    m3 = sm.OLS(y_sub, X3).fit(cov_type="HC1")
    log.info(f"    + intensity covs:   era coef = {m3.params[-1]:+.4f} "
             f"(SE={m3.bse[-1]:.4f}, p={m3.pvalues[-1]:.4f})")
    placebo_rows.append({
        "outcome": outcome_col, "specification": "baseline_plus_intensity",
        "era_coef": m3.params[-1], "era_se": m3.bse[-1],
        "era_pval": m3.pvalues[-1], "n": len(df_sub),
    })

placebo_df = pd.DataFrame(placebo_rows)
placebo_df.to_csv(OUT_TABLES / "sensitivity_placebo_test.csv", index=False)
log.info(f"\n  Saved: {OUT_TABLES / 'sensitivity_placebo_test.csv'}")


# ===================================================================
# 3. Estimate gamma (exclusion restriction violation magnitude)
# ===================================================================
log.info("\n" + "=" * 60)
log.info("GAMMA ESTIMATION (exclusion restriction violation)")
log.info("=" * 60)
log.info("Among ranibizumab-only patients, the era effect IS gamma")
log.info("(since the drug is held constant across eras).")

N_BOOT = 2000
rng = np.random.RandomState(42)

gamma_rows = []

outcomes_gamma = {
    "va_change": {"data": rani, "label": "VA change"},
}
if "event_va70" in outcomes_placebo:
    outcomes_gamma["event_va70"] = outcomes_placebo["event_va70"]
if "event_va35" in outcomes_placebo:
    outcomes_gamma["event_va35"] = outcomes_placebo["event_va35"]

for outcome_col, info in outcomes_gamma.items():
    df_sub = info["data"]
    y_sub = df_sub[outcome_col].values
    z_sub = df_sub["era_post2013"].values
    X_base = df_sub[covariates_baseline].values
    X_int = df_sub[covariates_intensity].values
    n_sub = len(df_sub)

    log.info(f"\n  Outcome: {info['label']} (n={n_sub})")

    # --- gamma_total: era effect with baseline covariates only ---
    X_g1 = np.column_stack([np.ones(n_sub), X_base, z_sub])
    m_g1 = sm.OLS(y_sub, X_g1).fit(cov_type="HC1")
    gamma_total = m_g1.params[-1]
    gamma_total_se = m_g1.bse[-1]
    log.info(f"    gamma_total (baseline covs):    {gamma_total:+.4f} "
             f"(SE={gamma_total_se:.4f}, p={m_g1.pvalues[-1]:.4f})")

    # --- gamma_residual: era effect with baseline + intensity covariates ---
    X_g2 = np.column_stack([np.ones(n_sub), X_base, X_int, z_sub])
    m_g2 = sm.OLS(y_sub, X_g2).fit(cov_type="HC1")
    gamma_residual = m_g2.params[-1]
    gamma_residual_se = m_g2.bse[-1]
    log.info(f"    gamma_residual (+ intensity):   {gamma_residual:+.4f} "
             f"(SE={gamma_residual_se:.4f}, p={m_g2.pvalues[-1]:.4f})")

    gamma_through_intensity = gamma_total - gamma_residual
    log.info(f"    gamma through intensity:        {gamma_through_intensity:+.4f}")

    # --- Bootstrap for gamma_total ---
    log.info(f"    Bootstrapping ({N_BOOT} iterations)...")
    boot_gammas = np.zeros(N_BOOT)
    idx = np.arange(n_sub)
    for b in range(N_BOOT):
        bi = rng.choice(idx, size=n_sub, replace=True)
        y_b = y_sub[bi]
        X_b = X_g1[bi]
        try:
            m_b = sm.OLS(y_b, X_b).fit()
            boot_gammas[b] = m_b.params[-1]
        except Exception:
            boot_gammas[b] = np.nan

    boot_gammas = boot_gammas[~np.isnan(boot_gammas)]
    gamma_boot_se = np.std(boot_gammas, ddof=1)
    gamma_boot_ci_lo = np.percentile(boot_gammas, 2.5)
    gamma_boot_ci_hi = np.percentile(boot_gammas, 97.5)
    log.info(f"    Bootstrap SE:  {gamma_boot_se:.4f}")
    log.info(f"    Bootstrap 95% CI: [{gamma_boot_ci_lo:.4f}, {gamma_boot_ci_hi:.4f}]")

    gamma_rows.append({
        "outcome": outcome_col,
        "gamma_total": gamma_total,
        "gamma_total_se_hc1": gamma_total_se,
        "gamma_total_pval": m_g1.pvalues[-1],
        "gamma_residual": gamma_residual,
        "gamma_residual_se_hc1": gamma_residual_se,
        "gamma_residual_pval": m_g2.pvalues[-1],
        "gamma_through_intensity": gamma_through_intensity,
        "gamma_boot_se": gamma_boot_se,
        "gamma_boot_ci_lo": gamma_boot_ci_lo,
        "gamma_boot_ci_hi": gamma_boot_ci_hi,
        "n_ranibizumab": n_sub,
        "n_boot": len(boot_gammas),
    })

gamma_df = pd.DataFrame(gamma_rows)
gamma_df.to_csv(OUT_TABLES / "sensitivity_gamma_estimates.csv", index=False)
log.info(f"\n  Saved: {OUT_TABLES / 'sensitivity_gamma_estimates.csv'}")


# ===================================================================
# 4. CHR sensitivity analysis
# ===================================================================
log.info("\n" + "=" * 60)
log.info("CHR SENSITIVITY ANALYSIS")
log.info("=" * 60)
log.info("LATE(gamma) = (ITT - gamma) / first_stage")

# Define outcomes for the full cohort
outcomes_chr = {}

# VA change: full cohort
outcomes_chr["va_change"] = {
    "label": "VA change (letters)",
    "y": pt["va_change"].values,
    "z": Z, "d": D,
    "x_base": pt[covariates_baseline].values,
    "n": len(pt),
}

# VA >= 70: eligible subset
eligible_70 = pt[pt["cohort_eligible_va70"] == 1].dropna(subset=["event_va70"])
if len(eligible_70) > 0 and eligible_70["era_post2013"].nunique() == 2:
    outcomes_chr["event_va70"] = {
        "label": "Achieved VA >= 70",
        "y": eligible_70["event_va70"].values,
        "z": eligible_70["era_post2013"].values,
        "d": eligible_70["treatment"].values,
        "x_base": eligible_70[covariates_baseline].values,
        "n": len(eligible_70),
    }

# VA <= 35: eligible subset
eligible_35 = pt[pt["cohort_eligible_va35"] == 1].dropna(subset=["event_va35"])
if len(eligible_35) > 0 and eligible_35["era_post2013"].nunique() == 2:
    outcomes_chr["event_va35"] = {
        "label": "Deteriorated to VA <= 35",
        "y": eligible_35["event_va35"].values,
        "z": eligible_35["era_post2013"].values,
        "d": eligible_35["treatment"].values,
        "x_base": eligible_35[covariates_baseline].values,
        "n": len(eligible_35),
    }

chr_results = {}  # store for plotting

for outcome_col, info in outcomes_chr.items():
    y_oc = info["y"]
    z_oc = info["z"]
    d_oc = info["d"]
    n_oc = info["n"]

    log.info(f"\n  Outcome: {info['label']} (n={n_oc})")

    # --- Compute ITT and first_stage (Wald components) ---
    n0 = (z_oc == 0).sum()
    n1 = (z_oc == 1).sum()

    y_bar_0 = y_oc[z_oc == 0].mean()
    y_bar_1 = y_oc[z_oc == 1].mean()
    itt = y_bar_1 - y_bar_0

    d_bar_0 = d_oc[z_oc == 0].mean()
    d_bar_1 = d_oc[z_oc == 1].mean()
    first_stage = d_bar_1 - d_bar_0

    # Wald LATE at gamma=0
    wald_late = itt / first_stage

    # Delta-method SE for Wald
    var_y0 = y_oc[z_oc == 0].var() / n0
    var_y1 = y_oc[z_oc == 1].var() / n1
    var_d0 = d_oc[z_oc == 0].var() / n0
    var_d1 = d_oc[z_oc == 1].var() / n1
    var_itt = var_y0 + var_y1
    var_fs = var_d0 + var_d1
    wald_se = (1 / first_stage) * np.sqrt(
        var_itt + (wald_late ** 2) * var_fs
    )

    log.info(f"    ITT = {itt:.4f}")
    log.info(f"    First stage = {first_stage:.4f}")
    log.info(f"    Wald LATE (gamma=0) = {wald_late:.4f} (SE={wald_se:.4f})")

    # --- Get gamma estimate for this outcome ---
    gamma_row = gamma_df[gamma_df["outcome"] == outcome_col]
    if len(gamma_row) == 0:
        log.info(f"    No gamma estimate available — skipping CHR for {outcome_col}")
        continue
    gamma_hat = gamma_row["gamma_total"].values[0]
    gamma_se = gamma_row["gamma_boot_se"].values[0]
    gamma_ci_lo = gamma_row["gamma_boot_ci_lo"].values[0]
    gamma_ci_hi = gamma_row["gamma_boot_ci_hi"].values[0]

    # --- Build gamma grid ---
    grid_lo = min(gamma_hat - 3 * gamma_se, -abs(gamma_hat) * 1.5)
    grid_hi = max(gamma_hat + 3 * gamma_se, abs(gamma_hat) * 1.5)
    # Ensure 0 is in range
    grid_lo = min(grid_lo, -0.5)
    grid_hi = max(grid_hi, 0.5)
    gamma_grid = np.linspace(grid_lo, grid_hi, 200)
    # Insert 0 and gamma_hat exactly
    gamma_grid = np.sort(np.unique(np.concatenate([gamma_grid, [0.0, gamma_hat]])))

    # --- Compute LATE(gamma) for each grid point ---
    # LATE(gamma) = (ITT - gamma) / first_stage
    # SE via delta method: SE(gamma) = sqrt(var_itt + gamma_se^2) / first_stage
    # But gamma from the grid is treated as fixed for each point;
    # the SE is for the Wald estimate given that gamma value.
    # SE_LATE(gamma) = SE_ITT / |first_stage|  (since gamma is fixed at each point)
    se_itt = np.sqrt(var_itt)

    late_grid = (itt - gamma_grid) / first_stage
    # SE for each point: propagate only ITT and first-stage uncertainty
    se_grid = (1 / abs(first_stage)) * np.sqrt(
        var_itt + (late_grid ** 2) * var_fs
    )
    ci_lo_grid = late_grid - 1.96 * se_grid
    ci_hi_grid = late_grid + 1.96 * se_grid

    # --- Find breakdown value ---
    # The gamma at which 95% CI first includes zero
    # CI includes zero when ci_lo <= 0 <= ci_hi
    includes_zero = (ci_lo_grid <= 0) & (ci_hi_grid >= 0)

    if wald_late > 0:
        # LATE is positive at gamma=0, look for where CI first includes 0
        # as gamma increases (LATE decreases)
        breakdown_gamma = np.nan
        for i in range(len(gamma_grid)):
            if gamma_grid[i] >= 0 and includes_zero[i]:
                breakdown_gamma = gamma_grid[i]
                break
    else:
        # LATE is negative at gamma=0, look in the other direction
        breakdown_gamma = np.nan
        for i in range(len(gamma_grid) - 1, -1, -1):
            if gamma_grid[i] <= 0 and includes_zero[i]:
                breakdown_gamma = gamma_grid[i]
                break

    log.info(f"    gamma_hat = {gamma_hat:+.4f} (boot SE={gamma_se:.4f})")
    log.info(f"    LATE(gamma_hat) = {(itt - gamma_hat) / first_stage:+.4f}")
    log.info(f"    Breakdown gamma = {breakdown_gamma:+.4f}"
             if not np.isnan(breakdown_gamma) else
             "    Breakdown gamma = not found in grid")

    # Store results
    chr_results[outcome_col] = {
        "label": info["label"],
        "itt": itt, "first_stage": first_stage,
        "wald_late": wald_late, "wald_se": wald_se,
        "gamma_hat": gamma_hat, "gamma_se": gamma_se,
        "gamma_ci_lo": gamma_ci_lo, "gamma_ci_hi": gamma_ci_hi,
        "gamma_grid": gamma_grid,
        "late_grid": late_grid, "se_grid": se_grid,
        "ci_lo_grid": ci_lo_grid, "ci_hi_grid": ci_hi_grid,
        "breakdown_gamma": breakdown_gamma,
        "n": n_oc,
    }

# Save CHR results table
chr_table_rows = []
for oc, res in chr_results.items():
    late_at_gamma = (res["itt"] - res["gamma_hat"]) / res["first_stage"]
    chr_table_rows.append({
        "outcome": oc,
        "itt": res["itt"],
        "first_stage": res["first_stage"],
        "wald_late_gamma0": res["wald_late"],
        "wald_se": res["wald_se"],
        "gamma_hat": res["gamma_hat"],
        "gamma_se": res["gamma_se"],
        "gamma_ci_lo": res["gamma_ci_lo"],
        "gamma_ci_hi": res["gamma_ci_hi"],
        "late_at_gamma_hat": late_at_gamma,
        "breakdown_gamma": res["breakdown_gamma"],
        "n": res["n"],
    })

chr_table_df = pd.DataFrame(chr_table_rows)
chr_table_df.to_csv(OUT_TABLES / "sensitivity_chr_results.csv", index=False)
log.info(f"\n  Saved: {OUT_TABLES / 'sensitivity_chr_results.csv'}")


# ===================================================================
# 5. Union of confidence intervals (UCI)
# ===================================================================
log.info("\n" + "=" * 60)
log.info("UNION OF CONFIDENCE INTERVALS (UCI)")
log.info("=" * 60)
log.info("When gamma in [gamma_lo, gamma_hi], the UCI is a single CI")
log.info("valid for any gamma in the plausible range.")

uci_rows = []

for oc, res in chr_results.items():
    gamma_lo = res["gamma_ci_lo"]
    gamma_hi = res["gamma_ci_hi"]
    fs = res["first_stage"]
    itt_val = res["itt"]
    var_itt_val = (res["wald_se"] * abs(fs)) ** 2 - (res["wald_late"] ** 2) * (
        0  # approximate: ignore first-stage variance contribution for UCI
    )
    # More careful: recompute from Wald SE
    # SE_wald = (1/fs) * sqrt(var_itt + late^2 * var_fs)
    # For UCI, use the Wald SE at each boundary gamma

    # LATE at boundaries
    late_at_gamma_lo = (itt_val - gamma_lo) / fs
    late_at_gamma_hi = (itt_val - gamma_hi) / fs

    # SE at boundaries (recompute with delta method)
    # Use var_itt from the Wald computation
    z_oc = res.get("z", None)
    # Recompute from stored values
    wald_se_val = res["wald_se"]

    # Approximate: use same SE as at gamma=0 (conservative)
    # The SE changes slightly with LATE but the difference is small
    se_at_lo = wald_se_val
    se_at_hi = wald_se_val

    # LATE is decreasing in gamma (since first_stage > 0), so:
    # UCI_lo uses the smallest LATE (at gamma_hi) minus margin
    # UCI_hi uses the largest LATE (at gamma_lo) plus margin
    if fs > 0:
        uci_lo = late_at_gamma_hi - 1.96 * se_at_hi
        uci_hi = late_at_gamma_lo + 1.96 * se_at_lo
    else:
        uci_lo = late_at_gamma_lo - 1.96 * se_at_lo
        uci_hi = late_at_gamma_hi + 1.96 * se_at_hi

    # Standard CI at gamma=0 for comparison
    std_ci_lo = res["wald_late"] - 1.96 * wald_se_val
    std_ci_hi = res["wald_late"] + 1.96 * wald_se_val

    log.info(f"\n  Outcome: {res['label']}")
    log.info(f"    gamma range: [{gamma_lo:.4f}, {gamma_hi:.4f}]")
    log.info(f"    LATE range:  [{late_at_gamma_hi:.4f}, {late_at_gamma_lo:.4f}]")
    log.info(f"    Standard CI (gamma=0): [{std_ci_lo:.4f}, {std_ci_hi:.4f}]")
    log.info(f"    UCI:                   [{uci_lo:.4f}, {uci_hi:.4f}]")

    uci_rows.append({
        "outcome": oc,
        "gamma_lo": gamma_lo, "gamma_hi": gamma_hi,
        "late_at_gamma_lo": late_at_gamma_lo,
        "late_at_gamma_hi": late_at_gamma_hi,
        "uci_lo": uci_lo, "uci_hi": uci_hi,
        "std_ci_lo": std_ci_lo, "std_ci_hi": std_ci_hi,
        "uci_wider_than_std": (uci_hi - uci_lo) > (std_ci_hi - std_ci_lo),
    })

uci_df = pd.DataFrame(uci_rows)
uci_df.to_csv(OUT_TABLES / "sensitivity_uci.csv", index=False)
log.info(f"\n  Saved: {OUT_TABLES / 'sensitivity_uci.csv'}")


# ===================================================================
# 6. Local-to-zero adjustment
# ===================================================================
log.info("\n" + "=" * 60)
log.info("LOCAL-TO-ZERO ADJUSTMENT")
log.info("=" * 60)
log.info("Model gamma = delta/sqrt(n). Produces a single adjusted estimate.")

for oc, res in chr_results.items():
    gamma_hat = res["gamma_hat"]
    gamma_se = res["gamma_se"]
    fs = res["first_stage"]
    wald_late_val = res["wald_late"]
    wald_se_val = res["wald_se"]

    late_adj = wald_late_val - gamma_hat / fs
    se_adj = np.sqrt(wald_se_val ** 2 + (gamma_se / fs) ** 2)
    ci_lo_adj = late_adj - 1.96 * se_adj
    ci_hi_adj = late_adj + 1.96 * se_adj
    p_adj = 2 * stats.norm.sf(abs(late_adj / se_adj))

    res["late_adj"] = late_adj
    res["se_adj"] = se_adj
    res["ci_lo_adj"] = ci_lo_adj
    res["ci_hi_adj"] = ci_hi_adj
    res["p_adj"] = p_adj

    log.info(f"\n  Outcome: {res['label']}")
    log.info(f"    Standard LATE:  {wald_late_val:+.4f} (SE={wald_se_val:.4f})")
    log.info(f"    gamma_hat:      {gamma_hat:+.4f} (SE={gamma_se:.4f})")
    log.info(f"    Adjusted LATE:  {late_adj:+.4f} (SE={se_adj:.4f})")
    log.info(f"    Adjusted 95%CI: [{ci_lo_adj:.4f}, {ci_hi_adj:.4f}]")
    log.info(f"    Adjusted p:     {p_adj:.4f}")


# ===================================================================
# 7. Intensity-conditioned 2SLS (secondary cross-check)
# ===================================================================
log.info("\n" + "=" * 60)
log.info("INTENSITY-CONDITIONED 2SLS (cross-check)")
log.info("=" * 60)
log.info("Adding injection intensity covariates to 2SLS.")
log.info("Caveat: intensity is partly post-treatment.")

intensity_2sls_rows = []

for outcome_col, info in outcomes_chr.items():
    y_oc = info["y"]
    z_oc = info["z"]
    d_oc = info["d"]

    # Identify the correct subset of patients for intensity covariates
    if outcome_col == "va_change":
        pt_sub = pt
    elif outcome_col == "event_va70":
        pt_sub = eligible_70
    elif outcome_col == "event_va35":
        pt_sub = eligible_35
    else:
        continue

    x_base = pt_sub[covariates_baseline].values
    x_full = pt_sub[covariates_baseline + covariates_intensity].values

    log.info(f"\n  Outcome: {info['label']}")

    # Standard 2SLS (baseline only)
    iv_base = iv_2sls(y_oc, d_oc, z_oc, x=x_base)
    log.info(f"    2SLS (baseline covs):   LATE={iv_base['coef']:+.4f} "
             f"(SE={iv_base['se']:.4f}, p={iv_base['p']:.4f})")

    # Intensity-conditioned 2SLS
    iv_full = iv_2sls(y_oc, d_oc, z_oc, x=x_full)
    log.info(f"    2SLS (+ intensity):     LATE={iv_full['coef']:+.4f} "
             f"(SE={iv_full['se']:.4f}, p={iv_full['p']:.4f})")

    intensity_2sls_rows.append({
        "outcome": outcome_col,
        "spec": "baseline_covariates",
        "late": iv_base["coef"], "se": iv_base["se"],
        "p": iv_base["p"], "f_first": iv_base["f_first"], "n": iv_base["n"],
    })
    intensity_2sls_rows.append({
        "outcome": outcome_col,
        "spec": "baseline_plus_intensity",
        "late": iv_full["coef"], "se": iv_full["se"],
        "p": iv_full["p"], "f_first": iv_full["f_first"], "n": iv_full["n"],
    })

    # Store for summary
    if outcome_col in chr_results:
        chr_results[outcome_col]["late_2sls_base"] = iv_base["coef"]
        chr_results[outcome_col]["se_2sls_base"] = iv_base["se"]
        chr_results[outcome_col]["late_2sls_intensity"] = iv_full["coef"]
        chr_results[outcome_col]["se_2sls_intensity"] = iv_full["se"]


# ===================================================================
# 8. Summary table
# ===================================================================
log.info("\n" + "=" * 60)
log.info("SUMMARY TABLE")
log.info("=" * 60)

summary_rows = []
for oc, res in chr_results.items():
    row = {
        "outcome": oc,
        "label": res["label"],
        "n": res["n"],
        "wald_late_gamma0": res["wald_late"],
        "wald_se": res["wald_se"],
        "wald_ci": f"[{res['wald_late'] - 1.96 * res['wald_se']:.3f}, "
                   f"{res['wald_late'] + 1.96 * res['wald_se']:.3f}]",
        "gamma_hat": res["gamma_hat"],
        "gamma_se": res["gamma_se"],
        "gamma_ci": f"[{res['gamma_ci_lo']:.3f}, {res['gamma_ci_hi']:.3f}]",
        "adjusted_late": res.get("late_adj", np.nan),
        "adjusted_se": res.get("se_adj", np.nan),
        "adjusted_ci": (
            f"[{res['ci_lo_adj']:.3f}, {res['ci_hi_adj']:.3f}]"
            if "ci_lo_adj" in res else ""
        ),
        "adjusted_p": res.get("p_adj", np.nan),
        "breakdown_gamma": res["breakdown_gamma"],
    }

    # UCI
    uci_row = uci_df[uci_df["outcome"] == oc]
    if len(uci_row) > 0:
        row["uci"] = f"[{uci_row['uci_lo'].values[0]:.3f}, {uci_row['uci_hi'].values[0]:.3f}]"
    else:
        row["uci"] = ""

    # Intensity-conditioned
    row["late_2sls_intensity"] = res.get("late_2sls_intensity", np.nan)
    row["se_2sls_intensity"] = res.get("se_2sls_intensity", np.nan)

    summary_rows.append(row)

    log.info(f"\n  {res['label']}:")
    log.info(f"    Standard LATE (gamma=0): {res['wald_late']:+.4f} "
             f"(SE={res['wald_se']:.4f})")
    log.info(f"    gamma_hat:               {res['gamma_hat']:+.4f}")
    log.info(f"    Adjusted LATE:           {res.get('late_adj', np.nan):+.4f} "
             f"(SE={res.get('se_adj', np.nan):.4f})")
    log.info(f"    Breakdown gamma:         {res['breakdown_gamma']:.4f}"
             if not np.isnan(res["breakdown_gamma"]) else
             "    Breakdown gamma:         not found")
    log.info(f"    UCI:                     {row['uci']}")
    if not np.isnan(res.get("late_2sls_intensity", np.nan)):
        log.info(f"    Intensity-cond 2SLS:     "
                 f"{res['late_2sls_intensity']:+.4f} "
                 f"(SE={res['se_2sls_intensity']:.4f})")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_TABLES / "sensitivity_summary.csv", index=False)
log.info(f"\n  Saved: {OUT_TABLES / 'sensitivity_summary.csv'}")


# ===================================================================
# 9. Figures
# ===================================================================
log.info("\n" + "=" * 60)
log.info("GENERATING FIGURES")
log.info("=" * 60)

# --- Figure 1: Main CHR sensitivity plot (va_change) ---
if "va_change" in chr_results:
    res = chr_results["va_change"]
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(res["gamma_grid"], res["late_grid"], "b-", linewidth=2,
            label="LATE(γ)")
    ax.fill_between(res["gamma_grid"], res["ci_lo_grid"], res["ci_hi_grid"],
                    alpha=0.2, color="blue", label="95% CI")

    # Reference lines
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="darkgreen", linestyle="--", linewidth=1.2,
               label="γ = 0 (standard IV)")
    ax.axvline(res["gamma_hat"], color="red", linestyle="-.", linewidth=1.2,
               label=f"γ = {res['gamma_hat']:.2f} (empirical)")

    # Breakdown value
    if not np.isnan(res["breakdown_gamma"]):
        ax.axvline(res["breakdown_gamma"], color="orange", linestyle=":",
                   linewidth=1.2,
                   label=f"Breakdown γ = {res['breakdown_gamma']:.2f}")

    # Annotate gamma CI
    ax.axvspan(res["gamma_ci_lo"], res["gamma_ci_hi"],
               alpha=0.08, color="red", label="γ 95% CI")

    ax.set_xlabel("γ (direct effect of era on outcome)")
    ax.set_ylabel("LATE (ETDRS letters)")
    ax.set_title("CHR Sensitivity Analysis: VA Change")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "chr_sensitivity.png")
    plt.close(fig)
    log.info(f"  Saved: {OUT_FIGURES / 'chr_sensitivity.png'}")

# --- Figure 2: Era placebo chart ---
if len(placebo_df[placebo_df["outcome"] == "va_change"]) > 0:
    pdf = placebo_df[placebo_df["outcome"] == "va_change"].copy()
    spec_labels = {
        "unadjusted": "Unadjusted",
        "baseline_covariates": "+ Baseline\ncovariates",
        "baseline_plus_intensity": "+ Intensity\ncovariates",
    }
    pdf["spec_label"] = pdf["specification"].map(spec_labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(len(pdf))
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    bars = ax.bar(x_pos, pdf["era_coef"].values, yerr=1.96 * pdf["era_se"].values,
                  capsize=5, color=colors[:len(pdf)], edgecolor="black",
                  linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pdf["spec_label"].values, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Era coefficient (ETDRS letters)")
    ax.set_title("Era Placebo Test: Ranibizumab-Only Patients\n"
                 "(VA change outcome)")

    # Add p-values
    for i, (_, row) in enumerate(pdf.iterrows()):
        p_str = f"p={row['era_pval']:.3f}" if row["era_pval"] >= 0.001 else "p<0.001"
        y_offset = row["era_coef"] + 1.96 * row["era_se"] + 0.15
        ax.text(i, y_offset, p_str, ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "era_placebo.png")
    plt.close(fig)
    log.info(f"  Saved: {OUT_FIGURES / 'era_placebo.png'}")

# --- Figure 3: Multi-outcome panel ---
n_outcomes = len(chr_results)
if n_outcomes > 0:
    ncols = min(n_outcomes, 2)
    nrows = (n_outcomes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, (oc, res) in enumerate(chr_results.items()):
        row_idx = idx // ncols
        col_idx = idx % ncols
        ax = axes[row_idx][col_idx]

        ax.plot(res["gamma_grid"], res["late_grid"], "b-", linewidth=1.5)
        ax.fill_between(res["gamma_grid"], res["ci_lo_grid"], res["ci_hi_grid"],
                        alpha=0.2, color="blue")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="darkgreen", linestyle="--", linewidth=1)
        ax.axvline(res["gamma_hat"], color="red", linestyle="-.", linewidth=1)
        if not np.isnan(res["breakdown_gamma"]):
            ax.axvline(res["breakdown_gamma"], color="orange", linestyle=":",
                       linewidth=1)
        ax.axvspan(res["gamma_ci_lo"], res["gamma_ci_hi"],
                   alpha=0.08, color="red")
        ax.set_xlabel("γ")
        ax.set_ylabel("LATE")
        ax.set_title(res["label"], fontsize=10)

    # Hide unused axes
    for idx in range(n_outcomes, nrows * ncols):
        row_idx = idx // ncols
        col_idx = idx % ncols
        axes[row_idx][col_idx].set_visible(False)

    fig.suptitle("CHR Sensitivity Analysis: All Outcomes", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "multi_outcome_sensitivity.png",
                bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {OUT_FIGURES / 'multi_outcome_sensitivity.png'}")


# ===================================================================
# Done
# ===================================================================
log.info("\n" + "=" * 60)
log.info("SENSITIVITY ANALYSIS COMPLETE")
log.info("=" * 60)
log.info(f"Tables: {OUT_TABLES}")
log.info(f"Figures: {OUT_FIGURES}")
log.info(f"Log: {log_path}")
