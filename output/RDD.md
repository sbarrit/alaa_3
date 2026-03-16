# Fuzzy Regression Discontinuity Design: Technical Report

## 1. Causal Question

Among patients treated around the time aflibercept became available at Moorfields Eye Hospital (October 2013), did gaining access to aflibercept improve visual acuity outcomes compared to ranibizumab?

## 2. Design

### 2.1 Identification strategy

Aflibercept was introduced at MEH in October 2013. Before this date, all patients received ranibizumab; after, clinicians could prescribe either drug. This creates a discontinuity in treatment access that can serve as a natural experiment.

We implement this as a **fuzzy RDD**, where the treatment era (post-2013 vs pre-2013) serves as an instrument for aflibercept receipt:

- **Instrument (Z):** binary era indicator (1 = post-2013, 0 = pre-2013)
- **Treatment (D):** aflibercept (1) vs ranibizumab (0)
- **Outcomes (Y):** VA change from baseline, achieved VA >= 70, deteriorated to VA <= 35, last recorded VA

### 2.2 Estimand

The **Local Average Treatment Effect (LATE)** — the causal effect of aflibercept (vs ranibizumab) among compliers, i.e. patients whose drug assignment was determined by the era they were treated in.

### 2.3 Estimation

Because the dataset contains only a binary era variable (not actual calendar dates), the fuzzy RDD reduces to an **instrumental variables / Wald estimator**:

$$\text{LATE} = \frac{E[Y \mid Z=1] - E[Y \mid Z=0]}{E[D \mid Z=1] - E[D \mid Z=0]}$$

We estimate this via:
1. **Wald estimator** (simple ratio of means with delta-method SEs)
2. **Two-stage least squares (2SLS)** with HC1 robust standard errors, both unconditional and with covariates

Covariates: baseline VA, sex, age, induction completion, induction injection interval, ethnicity dummies.

## 3. Results

### 3.1 First stage: era -> aflibercept

| Specification | Z coefficient | SE | F-statistic | N |
|---|---|---|---|---|
| Unconditional | 0.870 | 0.005 | 30,402 | 7,802 |
| With covariates | 0.866 | 0.005 | 24,814 | 7,802 |

Pure instrument: Pre-2013: 0% aflibercept. Post-2013: 87% aflibercept. The first-stage F far exceeds the Stock-Yogo threshold of 10 for weak instruments.

### 3.2 Main estimates: VA change from baseline

| Estimator | LATE (ETDRS letters) | 95% CI | p-value |
|---|---|---|---|
| Wald (unadjusted) | +1.34 | [0.40, 2.27] | 0.005 |
| 2SLS (unconditional) | +1.34 | [0.40, 2.27] | 0.005 |
| 2SLS (with covariates) | +0.78 | [-0.19, 1.76] | 0.116 |
| OLS naive (unconditional) | +1.07 | [0.28, 1.86] | 0.008 |
| OLS naive (with covariates) | +0.87 | [0.06, 1.67] | 0.035 |

The unadjusted estimate suggests aflibercept improves VA by approximately 1.3 ETDRS letters relative to ranibizumab. After covariate adjustment, the point estimate shrinks to +0.78 letters and is no longer statistically significant at the 5% level.

### 3.3 Secondary outcomes

| Outcome | Spec. | LATE | 95% CI | p | N |
|---|---|---|---|---|---|
| Achieved VA >= 70 | unadjusted | -0.012 | [-0.041, 0.018] | 0.437 | 5,978 |
| Achieved VA >= 70 | covariates | -0.005 | [-0.034, 0.023] | 0.709 | 5,978 |
| Deteriorated to VA <= 35 | unadjusted | -0.051 | [-0.075, -0.027] | <0.001 | 6,453 |
| Deteriorated to VA <= 35 | covariates | -0.014 | [-0.038, 0.010] | 0.246 | 6,453 |
| Last VA | unadjusted | +2.09 | [0.98, 3.20] | <0.001 | 7,802 |
| Last VA | covariates | +0.78 | [-0.19, 1.76] | 0.116 | 7,802 |

The unadjusted VA <= 35 deterioration result (5 percentage-point reduction) is notable but, like the primary outcome, does not survive covariate adjustment. The pattern is consistent: raw era differences favour post-2013 patients, but adjusting for baseline characteristics attenuates effects toward the null.

### 3.4 Covariate balance

| Covariate | Pre-2013 | Post-2013 | SMD | p-value |
|---|---|---|---|---|
| Baseline VA | 54.3 | 55.0 | +0.04 | 0.074 |
| Age midpoint | 78.0 | 78.4 | +0.05 | 0.040 |
| Female | 62.9% | 60.0% | -0.06 | 0.010 |
| **Loaded induction** | **64.7%** | **90.4%** | **+0.65** | **<0.001** |
| **Induction interval** | **65.5 d** | **35.3 d** | **-0.47** | **<0.001** |
| Afro-Caribbean | 2.3% | 2.3% | 0.00 | 0.998 |
| South-East Asian | 7.5% | 9.0% | +0.05 | 0.016 |
| **Unknown ethnicity** | **17.0%** | **35.5%** | **+0.43** | **<0.001** |

Three covariates show large imbalances (SMD > 0.1): loaded induction completion, induction injection interval, and unknown ethnicity. The first two reflect a **protocol change** concurrent with aflibercept introduction (shift from pro re nata to treat-and-extend dosing). This is not mere covariate imbalance but a structural threat to the exclusion restriction.

### 3.5 Sensitivity analyses

**Covariate specification sensitivity:**

| Specification | LATE | SE | p-value |
|---|---|---|---|
| Unadjusted | +1.34 | 0.48 | 0.005 |
| Demographics only | +1.41 | 0.47 | 0.003 |
| Demographics + baseline VA | +1.64 | 0.46 | <0.001 |
| Full covariates | +0.78 | 0.50 | 0.116 |
| Full + follow-up years | +0.56 | 0.49 | 0.256 |

The estimate is sensitive to specification. Adding induction-related covariates (loaded, interval) is what drives the attenuation from +1.34 to +0.78, confirming that the protocol change — not the drug — explains much of the raw era difference.

**Follow-up restriction sensitivity:**

| Max follow-up | LATE | SE | p | N |
|---|---|---|---|---|
| 2 years | -0.42 | 0.62 | 0.502 | 4,193 |
| 3 years | -0.26 | 0.56 | 0.643 | 5,431 |
| 5 years | -0.47 | 0.51 | 0.362 | 6,981 |

Restricting to patients with comparable follow-up lengths reverses the sign of the estimate. This suggests the positive unadjusted effect may be partly an artefact of differential follow-up: pre-2013 patients had longer follow-up, accumulating more late-stage vision loss.

**Placebo test:**

Regressing baseline VA on the era indicator (controlling for demographics) yields a coefficient of +1.84 (p < 0.001). The instrument significantly predicts a pre-treatment variable, which **violates the exclusion restriction** and indicates that patient composition shifted across eras beyond what the instrument should capture.

## 4. Interpretation

The evidence for a causal effect of aflibercept on visual acuity is **weak and fragile**. The headline finding (+1.34 ETDRS letters, p = 0.005) does not survive any of:

- Covariate adjustment (p = 0.116)
- Follow-up length restriction (sign reversal)
- The placebo test (instrument predicts baseline VA)

Even the unadjusted point estimate of +1.34 letters is clinically marginal — a single ETDRS line is 5 letters. This aligns with RCT evidence (e.g., CATT, VIEW 1/2 trials) showing non-inferiority between the two drugs, with no clinically meaningful difference in visual outcomes.

The most informative finding is **negative**: the apparent era effect is driven primarily by the concurrent protocol change (treat-and-extend replacing pro re nata) and differential follow-up, not by the drug substitution itself.

## 5. Limitations

1. **Binary running variable.** The dataset provides only a pre/post-2013 indicator, not actual treatment initiation dates. This collapses the RDD to a Wald/IV estimator with no ability to exploit proximity to the cutoff, select bandwidths, or test for manipulation of the running variable (McCrary test).

2. **Exclusion restriction violation.** October 2013 marks not only the introduction of aflibercept but also a shift in dosing protocol (pro re nata to treat-and-extend) and induction adherence norms. The instrument captures the full era change, not just drug access. The large SMDs on loaded induction (+0.65) and injection interval (-0.47) confirm this.

3. **Monotonicity questionable.** While no patient could receive aflibercept pre-2013 (supporting one-sided non-compliance), the 590 post-2013 ranibizumab patients may include a mix of physician preference, contraindications, or supply constraints — the reasons are unobserved.

4. **Differential follow-up.** Pre-2013 patients have longer potential follow-up (up to 12 years vs 6.3 years), mechanically creating differences in late-stage outcomes. Restricting follow-up reverses the treatment effect estimate.

5. **De-identified age.** Age is provided as 10-year bands, not continuous, limiting covariate adjustment precision.

## 6. Future Directions

1. **Request calendar dates.** With actual treatment initiation dates (even month-year granularity), a proper local polynomial RDD could be implemented with bandwidth selection (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik), density tests, and donut-hole specifications excluding patients exactly at the cutoff. ==> not practical

2. **Fixed-horizon outcomes.** Define outcomes at a standardised time point (e.g., VA at 1 year, VA at 2 years) to eliminate differential follow-up confounding. The longitudinal visit structure in `cohort_visits.csv.gz` supports this.

3. **Regression kink design.** If the probability of aflibercept receipt increased gradually around Oct 2013 (rather than as a sharp jump), a regression kink design on the continuous date could identify the effect from the change in slope.

4. **Bounding approaches.** Given the exclusion restriction concern, partial identification / bounds (Conley, Hansen & Rossi, 2012) could quantify how large the exclusion restriction violation would need to be to nullify the result.

5. **Complement with Analysis I (AIPW) and III (CATE).** The RDD provides one identification strategy; triangulating with the propensity-score-based ATE on the post-2013 cohort and the heterogeneous treatment effect analysis would strengthen the overall evidence base.

