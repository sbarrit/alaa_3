"""
run_preprocess.py — Clean and curate the MEH AMD dataset.

Reads:  input/MEH_AMD_survivaloutcomes_database.csv
Writes: output/results/cohort_patient.parquet   (one row per patient, N=7802)
        output/results/cohort_visits.parquet     (one row per visit, N≈118K)
        output/tables/preprocessing_summary.csv  (QC counts at each step)
        output/logs/log_preprocess_<timestamp>.log

Usage:
    python3 code/scripts/run_preprocess.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "input" / "MEH_AMD_survivaloutcomes_database.csv"
OUT_RESULTS = ROOT / "output" / "results"
OUT_TABLES = ROOT / "output" / "tables"
OUT_LOGS = ROOT / "output" / "logs"

for d in [OUT_RESULTS, OUT_TABLES, OUT_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT_LOGS / f"log_preprocess_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QC tracker
# ---------------------------------------------------------------------------
qc = []


def qc_step(label: str, df: pd.DataFrame, id_col: str = "anon_id"):
    n_rows = len(df)
    n_patients = df[id_col].nunique()
    qc.append({"step": label, "n_rows": n_rows, "n_patients": n_patients})
    log.info(f"[{label}] rows={n_rows:,}  patients={n_patients:,}")


# ===================================================================
# 1.  Load raw data
# ===================================================================
log.info(f"Loading {RAW_CSV}")
df = pd.read_csv(RAW_CSV)
qc_step("01_raw_load", df)

# ===================================================================
# 2.  Drop redundant index columns
# ===================================================================
drop_cols = [c for c in ["Unnamed: 0", "X"] if c in df.columns]
df = df.drop(columns=drop_cols)
log.info(f"Dropped columns: {drop_cols}")

# ===================================================================
# 3.  Standardise column names and categorical values
# ===================================================================
df = df.rename(columns={
    "va_inj1": "va_baseline",
    "va_inj1_group": "va_baseline_group",
    "date_inj1": "era",
    "mean_inj_interval": "induction_interval",
})

# Treatment binary: 1 = aflibercept, 0 = ranibizumab
df["treatment"] = (df["regimen"] == "Aflibercept only").astype(int)
df["treatment_label"] = df["treatment"].map({1: "aflibercept", 0: "ranibizumab"})

# Era binary: 1 = post-2013, 0 = pre-2013
df["era_post2013"] = (df["era"] == "Post-2013").astype(int)

# Loaded binary: 1 = induction completed, 0 = not
df["loaded_binary"] = (df["loaded"] == "loaded").astype(int)

# Gender binary: 1 = female, 0 = male
df["female"] = (df["gender"] == "f").astype(int)

# Injection given binary (NaN = monitoring visit -> 0)
df["injection_given"] = df["injgiven"].fillna(0).astype(int)

# Time in years (convenience)
df["time_years"] = df["time"] / 365.25

qc_step("02_standardised", df)

# ===================================================================
# 4.  Handle missing values
# ===================================================================
# 4a. Missing VA at visit (1,725 rows across 947 patients)
#     These are visits where VA was not recorded. Drop these rows
#     because VA is the outcome — cannot impute it.
n_va_miss = df["va"].isna().sum()
log.info(f"Rows with missing VA: {n_va_miss}")
df = df.dropna(subset=["va"])
qc_step("03_drop_missing_va", df)

# 4b. Missing induction_interval (388 patients with only 1 injection)
#     These patients never had a second injection, so no interval exists.
#     Fill with median of the cohort (clinically: they dropped out early).
median_interval = df["induction_interval"].median()
n_interval_miss = df["induction_interval"].isna().sum()
log.info(
    f"Rows with missing induction_interval: {n_interval_miss} "
    f"(388 single-injection patients). Filling with median={median_interval:.0f}"
)
df["induction_interval"] = df["induction_interval"].fillna(median_interval)

qc_step("04_impute_interval", df)

# ===================================================================
# 5.  Ethnicity grouping (collapse small categories)
# ===================================================================
# Original: caucasian, unknown/other, se_asian, afrocarribean, mixed
# Keep as-is but create a simplified version for modelling
ethnicity_map = {
    "caucasian": "white",
    "se_asian": "south_east_asian",
    "afrocarribean": "afro_caribbean",
    "mixed": "other",
    "unknown/other": "unknown",
}
df["ethnicity_clean"] = df["ethnicity"].map(ethnicity_map)

# One-hot for modelling (white as reference)
ethnicity_dummies = pd.get_dummies(
    df["ethnicity_clean"], prefix="eth", drop_first=False, dtype=int
)
# Keep all but drop 'white' as reference category
for c in ethnicity_dummies.columns:
    df[c] = ethnicity_dummies[c]

qc_step("05_ethnicity", df)

# ===================================================================
# 6.  Age group — ordinal encoding
# ===================================================================
age_order = {"50-59": 0, "60-69": 1, "70-79": 2, ">80": 3}
df["age_ordinal"] = df["age_group"].map(age_order)
# Midpoint for continuous modelling
age_midpoint = {"50-59": 55, "60-69": 65, "70-79": 75, ">80": 85}
df["age_midpoint"] = df["age_group"].map(age_midpoint)

qc_step("06_age_encoding", df)

# ===================================================================
# 7.  Build patient-level (time-independent) table
# ===================================================================
# One row per patient with baseline characteristics + outcome summaries
patient_cols = [
    "anon_id", "gender", "female", "ethnicity", "ethnicity_clean",
    "age_group", "age_ordinal", "age_midpoint",
    "va_baseline", "va_baseline_group",
    "era", "era_post2013",
    "induction_interval", "loaded", "loaded_binary",
    "regimen", "treatment", "treatment_label",
    # Ethnicity dummies
    "eth_afro_caribbean", "eth_other", "eth_south_east_asian",
    "eth_unknown", "eth_white",
]
patient = df.drop_duplicates("anon_id")[patient_cols].copy()

# Add derived patient-level summaries from the visit data
visit_summary = df.groupby("anon_id").agg(
    n_visits=("time", "count"),
    n_injections=("injection_given", "sum"),
    max_injnum=("injnum", "max"),
    follow_up_days=("time", "max"),
    va_last=("va", "last"),  # last recorded VA (after sorting by time)
    va_max=("va", "max"),
    va_min=("va", "min"),
).reset_index()
visit_summary["follow_up_years"] = visit_summary["follow_up_days"] / 365.25

patient = patient.merge(visit_summary, on="anon_id", how="left")

# --- Outcome 1: time to VA >= 70 (among those starting < 70) ---------------
eligible_70 = patient[patient["va_baseline"] < 70]["anon_id"]
events_70 = (
    df[df["anon_id"].isin(eligible_70) & (df["va"] >= 70)]
    .sort_values("time")
    .drop_duplicates("anon_id", keep="first")[["anon_id", "time"]]
    .rename(columns={"time": "time_to_va70"})
)
events_70["event_va70"] = 1

# Censored: last observation time
censored_70 = (
    df[df["anon_id"].isin(eligible_70) & ~df["anon_id"].isin(events_70["anon_id"])]
    .sort_values("time")
    .drop_duplicates("anon_id", keep="last")[["anon_id", "time"]]
    .rename(columns={"time": "time_to_va70"})
)
censored_70["event_va70"] = 0

outcome_70 = pd.concat([events_70, censored_70], ignore_index=True)
patient = patient.merge(outcome_70, on="anon_id", how="left")

# --- Outcome 2: time to VA <= 35 (among those starting > 35) ---------------
eligible_35 = patient[patient["va_baseline"] > 35]["anon_id"]
events_35 = (
    df[df["anon_id"].isin(eligible_35) & (df["va"] <= 35)]
    .sort_values("time")
    .drop_duplicates("anon_id", keep="first")[["anon_id", "time"]]
    .rename(columns={"time": "time_to_va35"})
)
events_35["event_va35"] = 1

censored_35 = (
    df[df["anon_id"].isin(eligible_35) & ~df["anon_id"].isin(events_35["anon_id"])]
    .sort_values("time")
    .drop_duplicates("anon_id", keep="last")[["anon_id", "time"]]
    .rename(columns={"time": "time_to_va35"})
)
censored_35["event_va35"] = 0

outcome_35 = pd.concat([events_35, censored_35], ignore_index=True)
patient = patient.merge(outcome_35, on="anon_id", how="left")

# --- Outcome 3: VA change from baseline at last visit -----------------------
patient["va_change"] = patient["va_last"] - patient["va_baseline"]

# Convert survival times to years
for col in ["time_to_va70", "time_to_va35"]:
    ycol = col + "_years"
    patient[ycol] = patient[col] / 365.25

log.info(f"Patient table: {len(patient)} rows, {len(patient.columns)} columns")
log.info(f"  Outcome VA>=70: {len(eligible_70)} eligible, "
         f"{int(events_70['event_va70'].sum())} events "
         f"({100*events_70['event_va70'].sum()/len(eligible_70):.1f}%)")
log.info(f"  Outcome VA<=35: {len(eligible_35)} eligible, "
         f"{int(events_35['event_va35'].sum())} events "
         f"({100*events_35['event_va35'].sum()/len(eligible_35):.1f}%)")

# ===================================================================
# 8.  Define analysis subcohorts
# ===================================================================
# Flag the key subcohorts for downstream analysis
patient["cohort_post2013"] = patient["era_post2013"]  # ATE / CATE analyses
patient["cohort_eligible_va70"] = patient["anon_id"].isin(eligible_70).astype(int)
patient["cohort_eligible_va35"] = patient["anon_id"].isin(eligible_35).astype(int)

n_post = patient["cohort_post2013"].sum()
n_post_afli = ((patient["cohort_post2013"] == 1) & (patient["treatment"] == 1)).sum()
n_post_rani = ((patient["cohort_post2013"] == 1) & (patient["treatment"] == 0)).sum()
log.info(f"Post-2013 cohort: {n_post} patients "
         f"(aflibercept={n_post_afli}, ranibizumab={n_post_rani})")

# ===================================================================
# 9.  Save outputs
# ===================================================================
# Patient-level
patient_path = OUT_RESULTS / "cohort_patient.csv"
patient.to_csv(patient_path, index=False)
log.info(f"Saved patient table -> {patient_path}")

# Visit-level (keep as cleaned longitudinal data)
visit_path = OUT_RESULTS / "cohort_visits.csv.gz"
df.to_csv(visit_path, index=False, compression="gzip")
log.info(f"Saved visit table -> {visit_path}")

# QC summary
qc_df = pd.DataFrame(qc)
qc_path = OUT_TABLES / "preprocessing_summary.csv"
qc_df.to_csv(qc_path, index=False)
log.info(f"Saved QC summary -> {qc_path}")

# ===================================================================
# 10. Print final summary
# ===================================================================
log.info("=" * 60)
log.info("PREPROCESSING COMPLETE")
log.info("=" * 60)
log.info(f"Visit-level:   {len(df):,} rows x {len(df.columns)} cols")
log.info(f"Patient-level: {len(patient):,} rows x {len(patient.columns)} cols")
log.info(f"  Total patients:        {len(patient):,}")
log.info(f"  Female:                {patient['female'].sum():,} "
         f"({100*patient['female'].mean():.1f}%)")
log.info(f"  Post-2013:             {n_post:,}")
log.info(f"  Aflibercept:           {(patient['treatment']==1).sum():,}")
log.info(f"  Ranibizumab:           {(patient['treatment']==0).sum():,}")
log.info(f"  Median follow-up:      "
         f"{patient['follow_up_years'].median():.1f} years")
log.info(f"  Median baseline VA:    {patient['va_baseline'].median():.0f} letters")
log.info(f"Output files:")
log.info(f"  {patient_path}")
log.info(f"  {visit_path}")
log.info(f"  {qc_path}")
log.info(f"  {log_path}")
