"""
run_rdd.py — Fuzzy Regression Discontinuity Design

Causal question: Among patients treated around the moment aflibercept became
available (Oct 2013), did gaining access to aflibercept improve visual outcomes?

Design: Fuzzy RDD exploiting the October 2013 introduction of aflibercept at
Moorfields Eye Hospital as a natural experiment.
  - Running variable: treatment era (binary: post-2013 vs pre-2013)
  - Instrument Z: era indicator (post-2013 = 1)
  - Treatment D: aflibercept (1) vs ranibizumab (0)
  - Outcome Y: VA change from baseline (ETDRS letters)

Because the running variable is binary (actual calendar dates are not in the
dataset), the fuzzy RDD reduces to an IV/Wald estimator:

    LATE = [E(Y|Z=1) - E(Y|Z=0)] / [E(D|Z=1) - E(D|Z=0)]

This is equivalent to 2SLS with the era as the excluded instrument.

Outputs:
    output/results/rdd_results.csv          Main results table
    output/tables/rdd_first_stage.csv       First-stage regression
    output/tables/rdd_reduced_form.csv      Reduced-form (ITT)
    output/tables/rdd_second_stage.csv      2SLS / Wald estimates
    output/tables/rdd_balance.csv           Covariate balance around cutoff
    output/tables/rdd_sensitivity.csv       Bandwidth and covariate sensitivity
    output/logs/log_rdd_<timestamp>.log

Usage:
    python3 code/scripts/run_rdd.py
"""

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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
OUT_LOGS = ROOT / "output" / "logs"

for d in [OUT_RESULTS, OUT_TABLES, OUT_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT_LOGS / f"log_rdd_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ===================================================================
# Helper: manual 2SLS (since linearmodels not available)
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
    # Test that delta = 0 (joint test on z columns)
    n_excl = z.shape[1]
    r_matrix = np.zeros((n_excl, first_X.shape[1]))
    for i in range(n_excl):
        r_matrix[i, W.shape[1] + i] = 1
    f_test = first_model.f_test(r_matrix)
    f_first = float(f_test.fvalue)

    # --- Second stage: Y = W*beta + D_hat*tau + e ---
    second_X = np.column_stack([W, d_hat])
    second_ols = sm.OLS(y, second_X).fit()

    # Correct SEs: replace D_hat with D in residuals
    resid_corrected = y - second_X @ second_ols.params
    # Actually need to use D not D_hat for residuals
    actual_X = np.column_stack([W, d])
    resid_corrected = y - actual_X @ second_ols.params

    # Robust variance using corrected residuals
    bread = np.linalg.inv(second_X.T @ second_X)
    meat = np.zeros((second_X.shape[1], second_X.shape[1]))
    for i in range(n):
        xi = second_X[i:i+1].T
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
# 1. Load data
# ===================================================================
log.info("Loading preprocessed patient data")
pt = pd.read_csv(PATIENT_CSV)
log.info(f"Loaded {len(pt)} patients")

# Key variables
Z = pt["era_post2013"].values          # instrument
D = pt["treatment"].values             # endogenous treatment
Y_change = pt["va_change"].values      # outcome: VA change from baseline

# ===================================================================
# 2. Descriptive statistics by era
# ===================================================================
log.info("=" * 60)
log.info("DESCRIPTIVE STATISTICS BY ERA")
log.info("=" * 60)

desc_rows = []
for label, col in [("VA change", "va_change"), ("Baseline VA", "va_baseline"),
                    ("Age midpoint", "age_midpoint"), ("Female", "female"),
                    ("Loaded", "loaded_binary"), ("Induction interval", "induction_interval"),
                    ("Follow-up (years)", "follow_up_years"),
                    ("Aflibercept", "treatment")]:
    pre = pt[pt["era_post2013"] == 0][col]
    post = pt[pt["era_post2013"] == 1][col]
    t, p = stats.ttest_ind(pre, post, equal_var=False)
    desc_rows.append({
        "variable": label,
        "pre_2013_mean": pre.mean(), "pre_2013_sd": pre.std(),
        "post_2013_mean": post.mean(), "post_2013_sd": post.std(),
        "t_stat": t, "p_value": p,
    })
    log.info(f"  {label:25s}: pre={pre.mean():7.2f} (SD={pre.std():.2f})  "
             f"post={post.mean():7.2f} (SD={post.std():.2f})  p={p:.4f}")

desc_df = pd.DataFrame(desc_rows)

# ===================================================================
# 3. First stage: Z -> D
# ===================================================================
log.info("=" * 60)
log.info("FIRST STAGE: era -> aflibercept")
log.info("=" * 60)

# Unconditional
p_d_z0 = D[Z == 0].mean()
p_d_z1 = D[Z == 1].mean()
first_stage_jump = p_d_z1 - p_d_z0
log.info(f"  P(aflibercept | pre-2013)  = {p_d_z0:.4f}")
log.info(f"  P(aflibercept | post-2013) = {p_d_z1:.4f}")
log.info(f"  First-stage jump           = {first_stage_jump:.4f}")

# With controls
covariates = ["va_baseline", "female", "age_midpoint", "loaded_binary",
              "induction_interval", "eth_afro_caribbean", "eth_south_east_asian",
              "eth_unknown", "eth_other"]
X_controls = pt[covariates].values

first_X_unc = sm.add_constant(Z)
first_model_unc = sm.OLS(D, first_X_unc).fit(cov_type="HC1")
log.info(f"\n  First stage (unconditional):")
log.info(f"    Z coef = {first_model_unc.params[1]:.4f} "
         f"(SE={first_model_unc.bse[1]:.4f}, p={first_model_unc.pvalues[1]:.2e})")
log.info(f"    F-stat = {first_model_unc.fvalue:.1f}")

first_X_cond = np.column_stack([np.ones(len(Z)), X_controls, Z])
first_model_cond = sm.OLS(D, first_X_cond).fit(cov_type="HC1")
log.info(f"\n  First stage (with covariates):")
log.info(f"    Z coef = {first_model_cond.params[-1]:.4f} "
         f"(SE={first_model_cond.bse[-1]:.4f}, p={first_model_cond.pvalues[-1]:.2e})")
log.info(f"    F-stat on Z = {(first_model_cond.params[-1]/first_model_cond.bse[-1])**2:.1f}")

# Save first stage
first_stage_rows = []
first_stage_rows.append({
    "specification": "unconditional",
    "z_coef": first_model_unc.params[1], "z_se": first_model_unc.bse[1],
    "z_pval": first_model_unc.pvalues[1], "f_stat": first_model_unc.fvalue,
    "r2": first_model_unc.rsquared, "n": int(first_model_unc.nobs),
})
first_stage_rows.append({
    "specification": "with_covariates",
    "z_coef": first_model_cond.params[-1], "z_se": first_model_cond.bse[-1],
    "z_pval": first_model_cond.pvalues[-1],
    "f_stat": (first_model_cond.params[-1] / first_model_cond.bse[-1]) ** 2,
    "r2": first_model_cond.rsquared, "n": int(first_model_cond.nobs),
})

# ===================================================================
# 4. Reduced form (ITT): Z -> Y
# ===================================================================
log.info("=" * 60)
log.info("REDUCED FORM (ITT): era -> VA change")
log.info("=" * 60)

# Unconditional
itt_unc = sm.OLS(Y_change, sm.add_constant(Z)).fit(cov_type="HC1")
log.info(f"  ITT (unconditional): {itt_unc.params[1]:.3f} "
         f"(SE={itt_unc.bse[1]:.3f}, p={itt_unc.pvalues[1]:.4f})")

# With covariates
itt_X = np.column_stack([np.ones(len(Z)), X_controls, Z])
itt_cond = sm.OLS(Y_change, itt_X).fit(cov_type="HC1")
log.info(f"  ITT (with covariates): {itt_cond.params[-1]:.3f} "
         f"(SE={itt_cond.bse[-1]:.3f}, p={itt_cond.pvalues[-1]:.4f})")

# ===================================================================
# 5. Wald estimator (simple ratio)
# ===================================================================
log.info("=" * 60)
log.info("WALD ESTIMATOR (LATE)")
log.info("=" * 60)

itt_effect = Y_change[Z == 1].mean() - Y_change[Z == 0].mean()
wald_late = itt_effect / first_stage_jump

# Delta method SE for Wald
n0, n1 = (Z == 0).sum(), (Z == 1).sum()
var_y0 = Y_change[Z == 0].var() / n0
var_y1 = Y_change[Z == 1].var() / n1
var_d0 = D[Z == 0].var() / n0  # = 0 since all pre-2013 are ranibizumab
var_d1 = D[Z == 1].var() / n1
# SE via delta method: SE(a/b) ≈ (1/b) * sqrt(var_a + (a/b)^2 * var_b)
var_num = var_y0 + var_y1
var_den = var_d0 + var_d1
wald_se = (1 / first_stage_jump) * np.sqrt(
    var_num + (wald_late ** 2) * var_den
)
wald_t = wald_late / wald_se
wald_p = 2 * stats.t.sf(abs(wald_t), df=len(Z) - 2)

log.info(f"  ITT (numerator):    {itt_effect:.3f}")
log.info(f"  First stage (den):  {first_stage_jump:.3f}")
log.info(f"  LATE (Wald):        {wald_late:.3f} ETDRS letters")
log.info(f"  SE (delta method):  {wald_se:.3f}")
log.info(f"  95% CI:             [{wald_late - 1.96*wald_se:.3f}, "
         f"{wald_late + 1.96*wald_se:.3f}]")
log.info(f"  p-value:            {wald_p:.4f}")

# ===================================================================
# 6. 2SLS estimation
# ===================================================================
log.info("=" * 60)
log.info("2SLS ESTIMATION")
log.info("=" * 60)

# 6a. Without covariates
iv_unc = iv_2sls(Y_change, D, Z)
log.info(f"  2SLS (unconditional):")
log.info(f"    LATE  = {iv_unc['coef']:.3f} (SE={iv_unc['se']:.3f})")
log.info(f"    95%CI = [{iv_unc['ci_lo']:.3f}, {iv_unc['ci_hi']:.3f}]")
log.info(f"    p     = {iv_unc['p']:.4f}")
log.info(f"    F(1st)= {iv_unc['f_first']:.1f}")

# 6b. With covariates
iv_cond = iv_2sls(Y_change, D, Z, x=X_controls)
log.info(f"\n  2SLS (with covariates):")
log.info(f"    LATE  = {iv_cond['coef']:.3f} (SE={iv_cond['se']:.3f})")
log.info(f"    95%CI = [{iv_cond['ci_lo']:.3f}, {iv_cond['ci_hi']:.3f}]")
log.info(f"    p     = {iv_cond['p']:.4f}")
log.info(f"    F(1st)= {iv_cond['f_first']:.1f}")

# ===================================================================
# 7. Multiple outcomes
# ===================================================================
log.info("=" * 60)
log.info("2SLS ACROSS OUTCOMES")
log.info("=" * 60)

outcome_results = []

# 7a. VA change (already done)
outcome_results.append({
    "outcome": "VA change (ETDRS letters)",
    "specification": "unconditional",
    "late": iv_unc["coef"], "se": iv_unc["se"],
    "ci_lo": iv_unc["ci_lo"], "ci_hi": iv_unc["ci_hi"],
    "p_value": iv_unc["p"], "f_first": iv_unc["f_first"], "n": iv_unc["n"],
})
outcome_results.append({
    "outcome": "VA change (ETDRS letters)",
    "specification": "with_covariates",
    "late": iv_cond["coef"], "se": iv_cond["se"],
    "ci_lo": iv_cond["ci_lo"], "ci_hi": iv_cond["ci_hi"],
    "p_value": iv_cond["p"], "f_first": iv_cond["f_first"], "n": iv_cond["n"],
})

# 7b. Binary: achieved VA >= 70 at any point (among eligible)
eligible_70 = pt[pt["cohort_eligible_va70"] == 1].copy()
if len(eligible_70) > 0:
    Y_70 = eligible_70["event_va70"].values
    Z_70 = eligible_70["era_post2013"].values
    D_70 = eligible_70["treatment"].values
    X_70 = eligible_70[covariates].values

    iv_70_unc = iv_2sls(Y_70, D_70, Z_70)
    iv_70_cond = iv_2sls(Y_70, D_70, Z_70, x=X_70)

    log.info(f"\n  Outcome: Achieved VA >= 70")
    log.info(f"    Unconditional: LATE={iv_70_unc['coef']:.4f} "
             f"(SE={iv_70_unc['se']:.4f}, p={iv_70_unc['p']:.4f})")
    log.info(f"    With covariates: LATE={iv_70_cond['coef']:.4f} "
             f"(SE={iv_70_cond['se']:.4f}, p={iv_70_cond['p']:.4f})")

    for spec, res in [("unconditional", iv_70_unc), ("with_covariates", iv_70_cond)]:
        outcome_results.append({
            "outcome": "Achieved VA >= 70",
            "specification": spec,
            "late": res["coef"], "se": res["se"],
            "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
            "p_value": res["p"], "f_first": res["f_first"], "n": res["n"],
        })

# 7c. Binary: deteriorated to VA <= 35 (among eligible)
eligible_35 = pt[pt["cohort_eligible_va35"] == 1].copy()
if len(eligible_35) > 0:
    Y_35 = eligible_35["event_va35"].values
    Z_35 = eligible_35["era_post2013"].values
    D_35 = eligible_35["treatment"].values
    X_35 = eligible_35[covariates].values

    iv_35_unc = iv_2sls(Y_35, D_35, Z_35)
    iv_35_cond = iv_2sls(Y_35, D_35, Z_35, x=X_35)

    log.info(f"\n  Outcome: Deteriorated to VA <= 35")
    log.info(f"    Unconditional: LATE={iv_35_unc['coef']:.4f} "
             f"(SE={iv_35_unc['se']:.4f}, p={iv_35_unc['p']:.4f})")
    log.info(f"    With covariates: LATE={iv_35_cond['coef']:.4f} "
             f"(SE={iv_35_cond['se']:.4f}, p={iv_35_cond['p']:.4f})")

    for spec, res in [("unconditional", iv_35_unc), ("with_covariates", iv_35_cond)]:
        outcome_results.append({
            "outcome": "Deteriorated to VA <= 35",
            "specification": spec,
            "late": res["coef"], "se": res["se"],
            "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
            "p_value": res["p"], "f_first": res["f_first"], "n": res["n"],
        })

# 7d. Last recorded VA
Y_last = pt["va_last"].values
iv_last_unc = iv_2sls(Y_last, D, Z)
iv_last_cond = iv_2sls(Y_last, D, Z, x=X_controls)
log.info(f"\n  Outcome: Last VA")
log.info(f"    Unconditional: LATE={iv_last_unc['coef']:.3f} "
         f"(SE={iv_last_unc['se']:.3f}, p={iv_last_unc['p']:.4f})")
log.info(f"    With covariates: LATE={iv_last_cond['coef']:.3f} "
         f"(SE={iv_last_cond['se']:.3f}, p={iv_last_cond['p']:.4f})")

for spec, res in [("unconditional", iv_last_unc), ("with_covariates", iv_last_cond)]:
    outcome_results.append({
        "outcome": "Last VA (ETDRS letters)",
        "specification": spec,
        "late": res["coef"], "se": res["se"],
        "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
        "p_value": res["p"], "f_first": res["f_first"], "n": res["n"],
    })

# ===================================================================
# 8. Covariate balance test
# ===================================================================
log.info("=" * 60)
log.info("COVARIATE BALANCE (validity check)")
log.info("=" * 60)
log.info("If era is as-if random near the cutoff, baseline covariates")
log.info("should not differ systematically between eras.")

balance_rows = []
for label, col in [("Baseline VA", "va_baseline"), ("Age midpoint", "age_midpoint"),
                    ("Female", "female"), ("Loaded induction", "loaded_binary"),
                    ("Induction interval (days)", "induction_interval"),
                    ("Afro-Caribbean", "eth_afro_caribbean"),
                    ("South-East Asian", "eth_south_east_asian"),
                    ("Unknown ethnicity", "eth_unknown")]:
    pre_vals = pt.loc[pt["era_post2013"] == 0, col]
    post_vals = pt.loc[pt["era_post2013"] == 1, col]
    pre_m, post_m = pre_vals.mean(), post_vals.mean()
    pooled_sd = np.sqrt((pre_vals.var() + post_vals.var()) / 2)
    smd = (post_m - pre_m) / pooled_sd if pooled_sd > 0 else 0
    t_val, p_val = stats.ttest_ind(pre_vals, post_vals, equal_var=False)
    balance_rows.append({
        "covariate": label,
        "pre_2013_mean": pre_m, "post_2013_mean": post_m,
        "smd": smd, "t_stat": t_val, "p_value": p_val,
    })
    flag = " ***" if abs(smd) > 0.1 else ""
    log.info(f"  {label:30s}: SMD={smd:+.3f}  p={p_val:.4f}{flag}")

balance_df = pd.DataFrame(balance_rows)

# ===================================================================
# 9. Sensitivity: covariate-adjusted specifications
# ===================================================================
log.info("=" * 60)
log.info("SENSITIVITY ANALYSES")
log.info("=" * 60)

sensitivity_rows = []

# 9a. Vary covariate sets
covariate_sets = {
    "unadjusted": [],
    "demographics": ["female", "age_midpoint"],
    "demographics_baseline_va": ["female", "age_midpoint", "va_baseline"],
    "full": covariates,
    "full_plus_followup": covariates + ["follow_up_years"],
}

for spec_name, spec_covs in covariate_sets.items():
    x = pt[spec_covs].values if spec_covs else None
    res = iv_2sls(Y_change, D, Z, x=x)
    sensitivity_rows.append({
        "analysis": "covariate_specification",
        "specification": spec_name,
        "late": res["coef"], "se": res["se"],
        "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
        "p_value": res["p"], "f_first": res["f_first"], "n": res["n"],
    })
    log.info(f"  {spec_name:35s}: LATE={res['coef']:+.3f} "
             f"(SE={res['se']:.3f}, p={res['p']:.4f})")

# 9b. Restrict to patients with matched follow-up lengths
# To address differential follow-up as a confounder
log.info("\n  --- Follow-up restriction sensitivity ---")
for max_fu_years in [2.0, 3.0, 5.0]:
    sub = pt[pt["follow_up_years"] <= max_fu_years]
    if sub["era_post2013"].nunique() < 2 or sub["treatment"].nunique() < 2:
        continue
    y_sub = sub["va_change"].values
    d_sub = sub["treatment"].values
    z_sub = sub["era_post2013"].values
    x_sub = sub[covariates].values
    res = iv_2sls(y_sub, d_sub, z_sub, x=x_sub)
    sensitivity_rows.append({
        "analysis": "followup_restriction",
        "specification": f"max_{max_fu_years:.0f}yr",
        "late": res["coef"], "se": res["se"],
        "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
        "p_value": res["p"], "f_first": res["f_first"], "n": res["n"],
    })
    log.info(f"  Follow-up <= {max_fu_years:.0f}yr (n={res['n']}): "
             f"LATE={res['coef']:+.3f} (SE={res['se']:.3f}, p={res['p']:.4f})")

# 9c. Placebo test: use instrument on pre-determined covariate (baseline VA)
# If era is a valid instrument, it should NOT predict baseline VA after
# controlling for other pre-treatment variables
log.info("\n  --- Placebo test: era -> baseline VA ---")
placebo_covs = ["female", "age_midpoint", "loaded_binary", "induction_interval",
                "eth_afro_caribbean", "eth_south_east_asian", "eth_unknown", "eth_other"]
placebo_X = np.column_stack([np.ones(len(Z)), pt[placebo_covs].values, Z])
placebo_model = sm.OLS(pt["va_baseline"].values, placebo_X).fit(cov_type="HC1")
placebo_coef = placebo_model.params[-1]
placebo_p = placebo_model.pvalues[-1]
log.info(f"  Era coef on baseline VA: {placebo_coef:.3f} (p={placebo_p:.4f})")
if placebo_p < 0.05:
    log.info("  ⚠ Significant — era predicts baseline VA, threatening validity")
else:
    log.info("  ✓ Not significant — supports instrument validity")

sensitivity_rows.append({
    "analysis": "placebo_test",
    "specification": "era_on_baseline_va",
    "late": placebo_coef, "se": placebo_model.bse[-1],
    "ci_lo": placebo_coef - 1.96 * placebo_model.bse[-1],
    "ci_hi": placebo_coef + 1.96 * placebo_model.bse[-1],
    "p_value": placebo_p, "f_first": np.nan, "n": len(Z),
})

sensitivity_df = pd.DataFrame(sensitivity_rows)

# ===================================================================
# 10. OLS comparison (naive)
# ===================================================================
log.info("=" * 60)
log.info("OLS COMPARISON (naive, biased)")
log.info("=" * 60)

# Naive OLS: Y = a + b*D + e
ols_unc = sm.OLS(Y_change, sm.add_constant(D)).fit(cov_type="HC1")
log.info(f"  OLS (unconditional): {ols_unc.params[1]:.3f} "
         f"(SE={ols_unc.bse[1]:.3f}, p={ols_unc.pvalues[1]:.4f})")

ols_X = np.column_stack([np.ones(len(D)), X_controls, D])
ols_cond = sm.OLS(Y_change, ols_X).fit(cov_type="HC1")
log.info(f"  OLS (with covariates): {ols_cond.params[-1]:.3f} "
         f"(SE={ols_cond.bse[-1]:.3f}, p={ols_cond.pvalues[-1]:.4f})")

# ===================================================================
# 11. Save all results
# ===================================================================
log.info("=" * 60)
log.info("SAVING RESULTS")
log.info("=" * 60)

# Main outcome results
outcome_df = pd.DataFrame(outcome_results)
outcome_df.to_csv(OUT_RESULTS / "rdd_results.csv", index=False)
log.info(f"  Saved: {OUT_RESULTS / 'rdd_results.csv'}")

# First stage
first_stage_df = pd.DataFrame(first_stage_rows)
first_stage_df.to_csv(OUT_TABLES / "rdd_first_stage.csv", index=False)

# Balance
balance_df.to_csv(OUT_TABLES / "rdd_balance.csv", index=False)

# Sensitivity
sensitivity_df.to_csv(OUT_TABLES / "rdd_sensitivity.csv", index=False)

# Combined summary for quick reference
summary_rows = [
    {"item": "Wald LATE (VA change)", "value": f"{wald_late:.2f}",
     "ci": f"[{wald_late-1.96*wald_se:.2f}, {wald_late+1.96*wald_se:.2f}]",
     "p": f"{wald_p:.4f}"},
    {"item": "2SLS LATE unconditional", "value": f"{iv_unc['coef']:.2f}",
     "ci": f"[{iv_unc['ci_lo']:.2f}, {iv_unc['ci_hi']:.2f}]",
     "p": f"{iv_unc['p']:.4f}"},
    {"item": "2SLS LATE with covariates", "value": f"{iv_cond['coef']:.2f}",
     "ci": f"[{iv_cond['ci_lo']:.2f}, {iv_cond['ci_hi']:.2f}]",
     "p": f"{iv_cond['p']:.4f}"},
    {"item": "First stage (unconditional)", "value": f"{first_stage_jump:.3f}",
     "ci": "", "p": f"{first_model_unc.pvalues[1]:.2e}"},
    {"item": "First stage F-stat", "value": f"{first_model_unc.fvalue:.1f}",
     "ci": "", "p": ""},
    {"item": "OLS naive (unconditional)", "value": f"{ols_unc.params[1]:.2f}",
     "ci": f"[{ols_unc.conf_int()[0][1]:.2f}, {ols_unc.conf_int()[1][1]:.2f}]",
     "p": f"{ols_unc.pvalues[1]:.4f}"},
    {"item": "OLS naive (with covariates)", "value": f"{ols_cond.params[-1]:.2f}",
     "ci": f"[{ols_cond.conf_int()[0][-1]:.2f}, {ols_cond.conf_int()[1][-1]:.2f}]",
     "p": f"{ols_cond.pvalues[-1]:.4f}"},
    {"item": "N total", "value": f"{len(pt)}", "ci": "", "p": ""},
    {"item": "N pre-2013", "value": f"{(Z==0).sum()}", "ci": "", "p": ""},
    {"item": "N post-2013", "value": f"{(Z==1).sum()}", "ci": "", "p": ""},
]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_TABLES / "rdd_summary.csv", index=False)
log.info(f"  Saved: {OUT_TABLES / 'rdd_summary.csv'}")

log.info("=" * 60)
log.info("RDD ANALYSIS COMPLETE")
log.info("=" * 60)
log.info(f"Log: {log_path}")
