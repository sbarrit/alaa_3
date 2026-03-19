"""
CATE Estimation for Anti-VEGF Drug Comparison in Neovascular AMD
================================================================
Estimates conditional average treatment effects (CATE) of aflibercept vs.
ranibizumab on the probability of achieving VA >= 70 within 2 years.

Uses the Moorfields Eye Hospital AMD dataset (Fu et al., 2020).

Analysis sample: All patients with baseline VA < 70 and >= 730 days of follow-up.
Treatment: Aflibercept (T=1) vs Ranibizumab (T=0)
Outcome: Binary — achieved VA >= 70 within 2 years (Y=1) or not (Y=0)

Methods:
  1. S-Learner (single model with treatment as feature)
  2. T-Learner (separate models per treatment arm)
  3. TARNet (Shalit et al., 2017 — shared representation + separate outcome heads)
  4. DragonNet (Shi et al., 2019 — TARNet + propensity-regularized representation)
  5. X-Learner (Künzel et al., 2019)
  6. IPW / AIPW as sanity checks

Author: Advaith Veturi (CPH200B Project)
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/claude')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

def load_and_prepare_data(filepath, time_horizon_days=730, restrict_post_2013=False):
    """
    Collapse longitudinal data to one row per patient.
    
    Parameters
    ----------
    filepath : str
    time_horizon_days : int
        Outcome horizon in days (730 = 2 years)
    restrict_post_2013 : bool
        If True, restrict to post-2013 patients only.
        If False, use all patients (better balance but era confounding).
    
    Returns
    -------
    df_patient : pd.DataFrame — one row per patient with covariates, treatment, outcome
    """
    df = pd.read_csv(filepath)
    
    if restrict_post_2013:
        df = df[df['date_inj1'] == 'Post-2013'].copy()
    
    # ----- Baseline covariates (one per patient) -----
    baseline = df.drop_duplicates('anon_id')[
        ['anon_id', 'gender', 'ethnicity', 'age_group', 'va_inj1', 
         'va_inj1_group', 'date_inj1', 'mean_inj_interval', 'loaded', 'regimen']
    ].copy()
    
    # Restrict to baseline VA < 70 (those who haven't already achieved the threshold)
    baseline = baseline[baseline['va_inj1'] < 70].copy()
    
    # ----- Determine outcome: reached VA >= 70 within time horizon -----
    eligible_ids = set(baseline['anon_id'])
    df_eligible = df[df['anon_id'].isin(eligible_ids)].copy()
    
    # Check max follow-up per patient
    max_follow = df_eligible.groupby('anon_id')['time'].max()
    
    # Require sufficient follow-up (>= time_horizon_days) OR event before horizon
    # First find who reached VA >= 70 and when
    def get_outcome(group):
        """For each patient, determine if VA >= 70 was reached within the horizon."""
        within_horizon = group[group['time'] <= time_horizon_days]
        reached = within_horizon[within_horizon['va'] >= 70]
        if len(reached) > 0:
            return pd.Series({
                'outcome': 1,
                'time_to_event': reached['time'].min(),
                'max_followup': group['time'].max(),
                'n_injections_2yr': len(within_horizon[within_horizon['injgiven'] == 1])
            })
        else:
            return pd.Series({
                'outcome': 0,
                'time_to_event': np.nan,
                'max_followup': group['time'].max(),
                'n_injections_2yr': len(within_horizon[within_horizon['injgiven'] == 1])
            })
    
    outcomes = df_eligible.groupby('anon_id').apply(get_outcome).reset_index()
    
    # Merge
    df_patient = baseline.merge(outcomes, on='anon_id', how='inner')
    
    # Filter: keep patients who either (a) reached the event, or (b) have follow-up >= horizon
    # This avoids censoring bias
    df_patient = df_patient[
        (df_patient['outcome'] == 1) | (df_patient['max_followup'] >= time_horizon_days)
    ].copy()
    
    # ----- Encode treatment -----
    df_patient['treatment'] = (df_patient['regimen'] == 'Aflibercept only').astype(int)
    
    # ----- Encode covariates -----
    df_patient['female'] = (df_patient['gender'] == 'f').astype(int)
    df_patient['loaded_binary'] = (df_patient['loaded'] == 'loaded').astype(int)
    df_patient['post_2013'] = (df_patient['date_inj1'] == 'Post-2013').astype(int)
    
    # Age group as ordinal
    age_map = {'50-59': 0, '60-69': 1, '70-79': 2, '>80': 3}
    df_patient['age_ordinal'] = df_patient['age_group'].map(age_map)
    
    # Impute missing mean_inj_interval with median
    median_interval = df_patient['mean_inj_interval'].median()
    df_patient['mean_inj_interval'] = df_patient['mean_inj_interval'].fillna(median_interval)
    
    # Ethnicity dummies
    eth_dummies = pd.get_dummies(df_patient['ethnicity'], prefix='eth', drop_first=True)
    df_patient = pd.concat([df_patient, eth_dummies], axis=1)
    
    return df_patient


def get_feature_matrix(df_patient):
    """Extract feature matrix X, treatment T, outcome Y."""
    feature_cols = [
        'va_inj1', 'age_ordinal', 'female', 'loaded_binary', 
        'mean_inj_interval', 'post_2013'
    ]
    # Add ethnicity dummies
    eth_cols = [c for c in df_patient.columns if c.startswith('eth_')]
    feature_cols += eth_cols
    
    X = df_patient[feature_cols].values.astype(float)
    T = df_patient['treatment'].values
    Y = df_patient['outcome'].values
    
    feature_names = feature_cols
    return X, T, Y, feature_names


# =============================================================================
# 2. PROPENSITY SCORE ESTIMATION
# =============================================================================

def estimate_propensity_scores(X, T, feature_names):
    """
    Estimate P(T=1|X) using logistic regression with cross-fitting.
    Also checks overlap and reports diagnostics.
    """
    print("\n" + "="*70)
    print("PROPENSITY SCORE ESTIMATION")
    print("="*70)
    
    # Cross-fitted propensity scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    ps_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    e_hat = cross_val_predict(ps_model, X_scaled, T, cv=cv, method='predict_proba')[:, 1]
    
    # Refit on full data for coefficient inspection
    ps_model.fit(X_scaled, T)
    
    print("\nPropensity model coefficients:")
    for name, coef in zip(feature_names, ps_model.coef_[0]):
        print(f"  {name:25s}: {coef:+.3f}")
    
    # Diagnostics
    print(f"\nPropensity score summary (T=1: aflibercept):")
    print(f"  T=0 (ranibizumab): mean={e_hat[T==0].mean():.3f}, "
          f"median={np.median(e_hat[T==0]):.3f}, "
          f"range=[{e_hat[T==0].min():.3f}, {e_hat[T==0].max():.3f}]")
    print(f"  T=1 (aflibercept): mean={e_hat[T==1].mean():.3f}, "
          f"median={np.median(e_hat[T==1]):.3f}, "
          f"range=[{e_hat[T==1].min():.3f}, {e_hat[T==1].max():.3f}]")
    
    # Overlap check
    trimmed = (e_hat > 0.05) & (e_hat < 0.95)
    print(f"\n  Patients with e(x) in [0.05, 0.95]: {trimmed.sum()} / {len(T)} "
          f"({100*trimmed.mean():.1f}%)")
    
    # AUC
    auc = roc_auc_score(T, e_hat)
    print(f"  Propensity model AUC: {auc:.3f}")
    
    return e_hat, scaler


# =============================================================================
# 3. CATE ESTIMATORS
# =============================================================================

class SLearner:
    """Single model with treatment as a feature."""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, T, Y):
        X_scaled = self.scaler.fit_transform(X)
        XT = np.column_stack([X_scaled, T])
        self.model.fit(XT, Y)
        return self
    
    def predict_cate(self, X):
        X_scaled = self.scaler.transform(X)
        X1 = np.column_stack([X_scaled, np.ones(len(X))])
        X0 = np.column_stack([X_scaled, np.zeros(len(X))])
        tau = self.model.predict_proba(X1)[:, 1] - self.model.predict_proba(X0)[:, 1]
        return tau


class TLearner:
    """Separate models per treatment arm."""
    
    def __init__(self):
        self.model_0 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.model_1 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, T, Y):
        X_scaled = self.scaler.fit_transform(X)
        self.model_0.fit(X_scaled[T == 0], Y[T == 0])
        self.model_1.fit(X_scaled[T == 1], Y[T == 1])
        return self
    
    def predict_cate(self, X):
        X_scaled = self.scaler.transform(X)
        mu1 = self.model_1.predict_proba(X_scaled)[:, 1]
        mu0 = self.model_0.predict_proba(X_scaled)[:, 1]
        return mu1 - mu0


class TARNetWrapper:
    def __init__(self):
        self.model = None
    
    def fit(self, X, T, Y):
        from nn_models import TARNet
        self.model = TARNet(input_dim=X.shape[1])
        self.model.fit(X, T, Y, verbose=False)
        return self
    
    def predict_cate(self, X):
        return self.model.predict_cate(X)


class DragonNetWrapper:
    def __init__(self):
        self.model = None
    
    def fit(self, X, T, Y):
        from nn_models import DragonNet
        self.model = DragonNet(input_dim=X.shape[1])
        self.model.fit(X, T, Y, verbose=False)
        return self
    
    def predict_cate(self, X):
        return self.model.predict_cate(X)


class XLearner:
    """
    X-Learner (Künzel et al., 2019).
    
    Stage 1: Fit outcome models per arm (T-learner).
    Stage 2: Impute individual treatment effects.
    Stage 3: Fit CATE models on imputed effects, combine with propensity weighting.
    """
    
    def __init__(self):
        # Stage 1: outcome models
        self.mu0_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.mu1_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        # Stage 2: CATE models
        self.tau0_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.tau1_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.scaler = StandardScaler()
        self.ps_model = LogisticRegression(max_iter=1000, C=1.0)
    
    def fit(self, X, T, Y):
        X_scaled = self.scaler.fit_transform(X)
        
        # Stage 1
        self.mu0_model.fit(X_scaled[T == 0], Y[T == 0])
        self.mu1_model.fit(X_scaled[T == 1], Y[T == 1])
        
        # Stage 2: impute treatment effects
        # For treated: D1 = Y1 - mu0(X)
        mu0_for_treated = self.mu0_model.predict_proba(X_scaled[T == 1])[:, 1]
        D1 = Y[T == 1] - mu0_for_treated
        
        # For control: D0 = mu1(X) - Y0
        mu1_for_control = self.mu1_model.predict_proba(X_scaled[T == 0])[:, 1]
        D0 = mu1_for_control - Y[T == 0]
        
        self.tau1_model.fit(X_scaled[T == 1], D1)
        self.tau0_model.fit(X_scaled[T == 0], D0)
        
        # Propensity for weighting
        self.ps_model.fit(X_scaled, T)
        
        return self
    
    def predict_cate(self, X):
        X_scaled = self.scaler.transform(X)
        
        tau1 = self.tau1_model.predict(X_scaled)
        tau0 = self.tau0_model.predict(X_scaled)
        
        # Weight by propensity
        e = self.ps_model.predict_proba(X_scaled)[:, 1]
        tau = e * tau0 + (1 - e) * tau1
        
        return tau


# =============================================================================
# 4. IPW-BASED ATE (SANITY CHECK)
# =============================================================================

def compute_ipw_ate(Y, T, e_hat, trim=0.05):
    """
    Inverse propensity weighting ATE estimate.
    Used as a sanity check — the mean of CATE estimates should roughly match.
    """
    mask = (e_hat > trim) & (e_hat < 1 - trim)
    Y_m, T_m, e_m = Y[mask], T[mask], e_hat[mask]
    
    ate = np.mean(Y_m * T_m / e_m - Y_m * (1 - T_m) / (1 - e_m))
    
    # Bootstrap CI
    n = len(Y_m)
    boot_ates = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        boot_ate = np.mean(
            Y_m[idx] * T_m[idx] / e_m[idx] - Y_m[idx] * (1 - T_m[idx]) / (1 - e_m[idx])
        )
        boot_ates.append(boot_ate)
    
    ci_low, ci_high = np.percentile(boot_ates, [2.5, 97.5])
    
    return ate, ci_low, ci_high


def compute_aipw_ate(Y, T, e_hat, mu0_hat, mu1_hat, trim=0.05):
    """
    Augmented IPW (doubly robust) ATE estimate.
    """
    mask = (e_hat > trim) & (e_hat < 1 - trim)
    Y_m = Y[mask]
    T_m = T[mask]
    e_m = e_hat[mask]
    mu0_m = mu0_hat[mask]
    mu1_m = mu1_hat[mask]
    
    # AIPW score
    score = (
        mu1_m - mu0_m
        + T_m * (Y_m - mu1_m) / e_m
        - (1 - T_m) * (Y_m - mu0_m) / (1 - e_m)
    )
    ate = np.mean(score)
    se = np.std(score) / np.sqrt(len(score))
    
    return ate, ate - 1.96*se, ate + 1.96*se


# =============================================================================
# 5. CROSS-VALIDATED CATE WITH BOOTSTRAP CIs
# =============================================================================

def cross_fitted_cate(X, T, Y, EstimatorClass, n_splits=5):
    """
    Cross-fitted CATE estimates to avoid overfitting.
    Returns out-of-fold CATE predictions for every observation.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tau_hat = np.zeros(len(Y))
    
    for train_idx, test_idx in cv.split(X, T):
        model = EstimatorClass()
        model.fit(X[train_idx], T[train_idx], Y[train_idx])
        tau_hat[test_idx] = model.predict_cate(X[test_idx])
    
    return tau_hat


# =============================================================================
# 6. VISUALIZATION
# =============================================================================

def plot_propensity_scores(e_hat, T, save_path=None):
    """Propensity score distribution by treatment arm."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(e_hat[T == 0], bins=50, alpha=0.6, label='Ranibizumab (T=0)', 
            density=True, color='#2196F3')
    ax.hist(e_hat[T == 1], bins=50, alpha=0.6, label='Aflibercept (T=1)', 
            density=True, color='#FF5722')
    ax.set_xlabel('Propensity Score P(Aflibercept | X)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Propensity Score Distributions by Treatment Arm', fontsize=14)
    ax.legend(fontsize=11)
    ax.axvline(0.05, ls='--', color='gray', alpha=0.5, label='Trim bounds')
    ax.axvline(0.95, ls='--', color='gray', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cate_distributions(cate_dict, save_path=None):
    """Distribution of CATE estimates across methods."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    
    for i, (name, tau) in enumerate(cate_dict.items()):
        ax.hist(tau, bins=50, alpha=0.5, label=f'{name} (mean={tau.mean():.3f})',
                density=True, color=colors[i % len(colors)])
    
    ax.axvline(0, ls='--', color='black', alpha=0.7, lw=1.5)
    ax.set_xlabel('CATE: P(VA≥70 | Aflibercept) − P(VA≥70 | Ranibizumab)', fontsize=11)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Estimated CATE Across Methods', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cate_by_subgroup(df_patient, tau_hat, method_name, save_path=None):
    """CATE estimates stratified by baseline VA and age group."""
    df_plot = df_patient.copy()
    df_plot['tau_hat'] = tau_hat
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # By baseline VA bins
    df_plot['va_bin'] = pd.cut(df_plot['va_inj1'], bins=[0, 35, 49, 69], 
                                labels=['≤35', '36-49', '50-69'])
    
    ax = axes[0]
    va_groups = df_plot.groupby('va_bin')['tau_hat']
    means = va_groups.mean()
    sems = va_groups.sem()
    x_pos = range(len(means))
    ax.bar(x_pos, means, yerr=1.96*sems, capsize=5, color=['#BBDEFB', '#64B5F6', '#1976D2'],
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(means.index)
    ax.axhline(0, ls='--', color='black', alpha=0.5)
    ax.set_xlabel('Baseline VA Group', fontsize=12)
    ax.set_ylabel('Mean CATE', fontsize=12)
    ax.set_title('CATE by Baseline Visual Acuity', fontsize=13)
    
    # By age group
    ax = axes[1]
    age_order = ['50-59', '60-69', '70-79', '>80']
    df_plot['age_group_ordered'] = pd.Categorical(df_plot['age_group'], 
                                                    categories=age_order, ordered=True)
    age_groups = df_plot.groupby('age_group_ordered')['tau_hat']
    means = age_groups.mean()
    sems = age_groups.sem()
    x_pos = range(len(means))
    ax.bar(x_pos, means, yerr=1.96*sems, capsize=5, 
           color=['#C8E6C9', '#81C784', '#43A047', '#1B5E20'],
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(means.index)
    ax.axhline(0, ls='--', color='black', alpha=0.5)
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Mean CATE', fontsize=12)
    ax.set_title('CATE by Age at Baseline', fontsize=13)
    
    # By loading status
    ax = axes[2]
    load_groups = df_plot.groupby('loaded')['tau_hat']
    means = load_groups.mean()
    sems = load_groups.sem()
    x_pos = range(len(means))
    ax.bar(x_pos, means, yerr=1.96*sems, capsize=5,
           color=['#FFCC80', '#FF9800'], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(means.index)
    ax.axhline(0, ls='--', color='black', alpha=0.5)
    ax.set_xlabel('Induction Phase', fontsize=12)
    ax.set_ylabel('Mean CATE', fontsize=12)
    ax.set_title('CATE by Induction Completion', fontsize=13)
    
    fig.suptitle(f'Heterogeneous Treatment Effects — {method_name}', fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cate_heatmap(df_patient, tau_hat, method_name, save_path=None):
    """
    CATE heatmap over baseline VA × age group — the clinically actionable output.
    """
    df_plot = df_patient.copy()
    df_plot['tau_hat'] = tau_hat
    
    # Create VA bins
    df_plot['va_bin'] = pd.cut(df_plot['va_inj1'], bins=[0, 35, 49, 69],
                                labels=['≤35\n(20/200)', '36-49\n(20/200-20/100)', 
                                        '50-69\n(20/100-20/40)'])
    
    age_order = ['50-59', '60-69', '70-79', '>80']
    
    # Pivot table
    pivot = df_plot.pivot_table(values='tau_hat', index='va_bin', columns='age_group',
                                 aggfunc='mean')
    pivot = pivot[age_order]
    
    # Count table for annotation
    count_pivot = df_plot.pivot_table(values='tau_hat', index='va_bin', columns='age_group',
                                       aggfunc='count')
    count_pivot = count_pivot[age_order]
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    
    # Create annotation strings as a numpy array of strings
    annot_arr = np.empty(pivot.shape, dtype=object)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            n = count_pivot.iloc[i, j] if not pd.isna(count_pivot.iloc[i, j]) else 0
            if pd.isna(val):
                annot_arr[i, j] = ''
            else:
                annot_arr[i, j] = f'{val:+.3f}\n(n={int(n)})'
    
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    
    sns.heatmap(pivot, annot=annot_arr, fmt='', cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax, ax=ax, linewidths=1, linecolor='white',
                cbar_kws={'label': 'CATE (Aflibercept − Ranibizumab)'})
    
    ax.set_xlabel('Age Group at Baseline', fontsize=12)
    ax.set_ylabel('Baseline Visual Acuity', fontsize=12)
    ax.set_title(f'CATE Heatmap: Aflibercept vs Ranibizumab Effect on P(VA≥70)\n'
                 f'Method: {method_name}', fontsize=13)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_treatment_rule(df_patient, tau_hat, method_name, save_path=None):
    """
    Visualize the implied treatment decision rule.
    """
    df_plot = df_patient.copy()
    df_plot['tau_hat'] = tau_hat
    df_plot['recommended'] = np.where(tau_hat > 0, 'Aflibercept', 'Ranibizumab')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: CATE vs baseline VA, colored by recommendation
    ax = axes[0]
    colors = np.where(tau_hat > 0, '#FF5722', '#2196F3')
    ax.scatter(df_plot['va_inj1'], tau_hat, c=colors, alpha=0.3, s=10)
    ax.axhline(0, ls='--', color='black', lw=1.5)
    ax.set_xlabel('Baseline VA (ETDRS letter score)', fontsize=12)
    ax.set_ylabel('Estimated CATE', fontsize=12)
    ax.set_title('CATE vs Baseline VA', fontsize=13)
    
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF5722', alpha=0.6, label='Recommend Aflibercept'),
        Patch(facecolor='#2196F3', alpha=0.6, label='Recommend Ranibizumab')
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    
    # Pie chart of recommendations
    ax = axes[1]
    rec_counts = df_plot['recommended'].value_counts()
    ax.pie(rec_counts, labels=rec_counts.index, autopct='%1.1f%%',
           colors=['#FF5722', '#2196F3'], startangle=90,
           textprops={'fontsize': 12})
    ax.set_title(f'Treatment Recommendations\n({method_name})', fontsize=13)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration_check(Y, T, tau_hat, e_hat, method_name, n_bins=5, save_path=None):
    """
    CATE calibration: bin by predicted CATE, compare observed treatment effect within bins.
    """
    df_cal = pd.DataFrame({
        'Y': Y, 'T': T, 'tau_hat': tau_hat, 'e_hat': e_hat
    })
    
    # Trim for overlap
    df_cal = df_cal[(df_cal['e_hat'] > 0.05) & (df_cal['e_hat'] < 0.95)]
    
    # Create CATE bins
    df_cal['tau_bin'] = pd.qcut(df_cal['tau_hat'], q=n_bins, duplicates='drop')
    
    results = []
    for name, group in df_cal.groupby('tau_bin'):
        predicted = group['tau_hat'].mean()
        # IPW estimate within bin
        t1 = group[group['T'] == 1]
        t0 = group[group['T'] == 0]
        if len(t1) > 5 and len(t0) > 5:
            # Simple difference in means (crude but interpretable)
            observed = t1['Y'].mean() - t0['Y'].mean()
            results.append({
                'bin': str(name),
                'predicted_cate': predicted,
                'observed_diff': observed,
                'n': len(group)
            })
    
    if len(results) < 2:
        print(f"  [WARNING] Not enough bins with both arms for calibration check")
        return
    
    res_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(res_df['predicted_cate'], res_df['observed_diff'], s=res_df['n']/2,
               color='#1976D2', edgecolors='black', linewidth=0.5, zorder=5)
    
    # 45-degree line
    lims = [min(res_df['predicted_cate'].min(), res_df['observed_diff'].min()) - 0.02,
            max(res_df['predicted_cate'].max(), res_df['observed_diff'].max()) + 0.02]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect calibration')
    
    ax.set_xlabel('Predicted CATE (binned mean)', fontsize=12)
    ax.set_ylabel('Observed treatment effect (within bin)', fontsize=12)
    ax.set_title(f'CATE Calibration Check — {method_name}', fontsize=13)
    ax.legend(fontsize=10)
    
    for _, row in res_df.iterrows():
        ax.annotate(f'n={int(row["n"])}', (row['predicted_cate'], row['observed_diff']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# 7. MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("CATE ESTIMATION: AFLIBERCEPT vs RANIBIZUMAB")
    print("Outcome: P(VA ≥ 70 within 2 years)")
    print("=" * 70)
    
    # --- Load data ---
    # Using ALL patients (not restricted to post-2013) for better treatment balance.
    # The post-2013 indicator is included as a covariate to control for era effects.
    # Sensitivity analysis with post-2013 restriction follows.
    
    print("\n>>> PRIMARY ANALYSIS: Full sample (era as covariate)")
    df = load_and_prepare_data(
        'doi_10_5061_dryad_nvx0k6dqg__v20201105/MEH_AMD_survivaloutcomes_database.csv',
        time_horizon_days=730,
        restrict_post_2013=True
    )
    
    print(f"\nAnalysis sample: {len(df)} patients")
    print(f"  Aflibercept: {(df['treatment']==1).sum()}")
    print(f"  Ranibizumab: {(df['treatment']==0).sum()}")
    print(f"  Reached VA≥70 within 2yr: {df['outcome'].sum()} ({100*df['outcome'].mean():.1f}%)")
    print(f"  By treatment:")
    for t, name in [(1, 'Aflibercept'), (0, 'Ranibizumab')]:
        sub = df[df['treatment'] == t]
        print(f"    {name}: {sub['outcome'].mean():.3f} ({sub['outcome'].sum()}/{len(sub)})")
    
    X, T, Y, feature_names = get_feature_matrix(df)
    
    # --- Propensity scores ---
    e_hat, scaler = estimate_propensity_scores(X, T, feature_names)
    plot_propensity_scores(e_hat, T, save_path='results/fig_propensity.png')
    
    # --- Naive ATE ---
    print("\n" + "="*70)
    print("NAIVE & IPW ATE ESTIMATES (SANITY CHECKS)")
    print("="*70)
    
    naive_ate = Y[T==1].mean() - Y[T==0].mean()
    print(f"\n  Naive difference in means: {naive_ate:+.4f}")
    
    ipw_ate, ipw_lo, ipw_hi = compute_ipw_ate(Y, T, e_hat)
    print(f"  IPW ATE: {ipw_ate:+.4f} (95% CI: [{ipw_lo:+.4f}, {ipw_hi:+.4f}])")
    
    # AIPW using T-learner predictions
    print("\n  Fitting T-learner for AIPW outcome models...")
    tl_temp = TLearner()
    tl_temp.fit(X, T, Y)
    X_scaled_temp = tl_temp.scaler.transform(X)
    mu0_hat = tl_temp.model_0.predict_proba(X_scaled_temp)[:, 1]
    mu1_hat = tl_temp.model_1.predict_proba(X_scaled_temp)[:, 1]
    
    aipw_ate, aipw_lo, aipw_hi = compute_aipw_ate(Y, T, e_hat, mu0_hat, mu1_hat)
    print(f"  AIPW ATE: {aipw_ate:+.4f} (95% CI: [{aipw_lo:+.4f}, {aipw_hi:+.4f}])")
    
    # --- CATE estimation ---
    print("\n" + "="*70)
    print("CATE ESTIMATION (CROSS-FITTED)")
    print("="*70)
    
    estimators = {
        'S-Learner': SLearner,
        'T-Learner': TLearner,
        'TARNet': TARNetWrapper,
        'DragonNet': DragonNetWrapper,
        'X-Learner': XLearner,
    }
    
    cate_results = {}
    for name, EstClass in estimators.items():
        print(f"\n  Fitting {name}...")
        tau = cross_fitted_cate(X, T, Y, EstClass, n_splits=5)
        cate_results[name] = tau
        print(f"    Mean CATE: {tau.mean():+.4f}")
        print(f"    Std CATE:  {tau.std():.4f}")
        print(f"    Range:     [{tau.min():+.4f}, {tau.max():+.4f}]")
        print(f"    % favoring aflibercept: {100*(tau > 0).mean():.1f}%")
    
    # --- ATE consistency check ---
    print("\n" + "="*70)
    print("ATE CONSISTENCY CHECK")
    print("="*70)
    print(f"\n  {'Method':<20s} {'Mean CATE':>10s}")
    print(f"  {'-'*30}")
    print(f"  {'Naive':.<20s} {naive_ate:>+10.4f}")
    print(f"  {'IPW':.<20s} {ipw_ate:>+10.4f}")
    print(f"  {'AIPW':.<20s} {aipw_ate:>+10.4f}")
    for name, tau in cate_results.items():
        print(f"  {name:.<20s} {tau.mean():>+10.4f}")
    
    # --- Plots ---
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # CATE distributions
    plot_cate_distributions(cate_results, save_path='results/fig_cate_distributions.png')
    print("  Saved: fig_cate_distributions.png")
    
    # Use X-Learner as primary (generally best for imbalanced treatment)
    primary_method = 'X-Learner'
    primary_tau = cate_results[primary_method]
    
    # Subgroup effects
    plot_cate_by_subgroup(df, primary_tau, primary_method, save_path='results/fig_cate_subgroups.png')
    print("  Saved: fig_cate_subgroups.png")
    
    # Heatmap
    plot_cate_heatmap(df, primary_tau, primary_method, save_path='results/fig_cate_heatmap.png')
    print("  Saved: fig_cate_heatmap.png")
    
    # Also do heatmaps for other methods for comparison
    for name, tau in cate_results.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        plot_cate_heatmap(df, tau, name, save_path=f'results/fig_heatmap_{safe_name}.png')
    print("  Saved: heatmaps for all methods")
    
    # Treatment rule
    plot_treatment_rule(df, primary_tau, primary_method, save_path='results/fig_treatment_rule.png')
    print("  Saved: fig_treatment_rule.png")
    
    # Calibration check
    plot_calibration_check(Y, T, primary_tau, e_hat, primary_method,
                           save_path='results/fig_calibration.png')
    print("  Saved: fig_calibration.png")
    
    # --- Summary table ---
    print("\n" + "="*70)
    print("CATE SUMMARY BY SUBGROUP (X-Learner)")
    print("="*70)
    
    df['tau_hat'] = primary_tau
    
    print(f"\n  {'Subgroup':<35s} {'Mean CATE':>10s} {'N':>6s}")
    print(f"  {'-'*51}")
    
    for age in ['50-59', '60-69', '70-79', '>80']:
        sub = df[df['age_group'] == age]
        print(f"  Age {age:<30s} {sub['tau_hat'].mean():>+10.4f} {len(sub):>6d}")
    
    print()
    for va_grp, label in [((0, 36), 'VA ≤ 35'), ((36, 50), 'VA 36-49'), ((50, 70), 'VA 50-69')]:
        sub = df[(df['va_inj1'] >= va_grp[0]) & (df['va_inj1'] < va_grp[1])]
        print(f"  {label:<35s} {sub['tau_hat'].mean():>+10.4f} {len(sub):>6d}")
    
    print()
    for loaded_val, label in [('loaded', 'Loaded (induction complete)'), 
                               ('notloaded', 'Not loaded')]:
        sub = df[df['loaded'] == loaded_val]
        print(f"  {label:<35s} {sub['tau_hat'].mean():>+10.4f} {len(sub):>6d}")
    
    # --- Treatment rule value ---
    print("\n" + "="*70)
    print("TREATMENT RULE VALUE ESTIMATION")
    print("="*70)
    
    n_recommend_aflib = (primary_tau > 0).sum()
    n_recommend_ranib = (primary_tau <= 0).sum()
    print(f"\n  Patients recommended aflibercept: {n_recommend_aflib} "
          f"({100*n_recommend_aflib/len(primary_tau):.1f}%)")
    print(f"  Patients recommended ranibizumab: {n_recommend_ranib} "
          f"({100*n_recommend_ranib/len(primary_tau):.1f}%)")
    print(f"\n  If all these patients could switch to the recommended drug,")
    print(f"  the expected improvement in P(VA≥70) would be: "
          f"{np.mean(np.abs(primary_tau)):.4f}")
    
    # =========================================================================
    # SENSITIVITY ANALYSIS: Post-2013 only
    # =========================================================================
    print("\n\n" + "="*70)
    print("SENSITIVITY ANALYSIS: POST-2013 PATIENTS ONLY")
    print("="*70)
    
    df_post = load_and_prepare_data(
        'doi_10_5061_dryad_nvx0k6dqg__v20201105/MEH_AMD_survivaloutcomes_database.csv',
        time_horizon_days=730,
        restrict_post_2013=True
    )
    
    print(f"\nPost-2013 analysis sample: {len(df_post)} patients")
    print(f"  Aflibercept: {(df_post['treatment']==1).sum()}")
    print(f"  Ranibizumab: {(df_post['treatment']==0).sum()}")
    
    X_post, T_post, Y_post, fn_post = get_feature_matrix(df_post)
    
    # Note: severe imbalance — flag this
    imbalance_ratio = (T_post == 1).sum() / (T_post == 0).sum()
    print(f"\n  WARNING: Treatment ratio (AFB/RBZ) = {imbalance_ratio:.1f}:1")
    print(f"  This severe imbalance limits the reliability of CATE estimates.")
    print(f"  The primary analysis (full sample) is preferred.\n")
    
    if (T_post == 0).sum() >= 50:  # Only run if enough control patients
        e_post, _ = estimate_propensity_scores(X_post, T_post, fn_post)
        
        for name, EstClass in {'X-Learner': XLearner, 'T-Learner': TLearner, 'DragonNet': DragonNetWrapper}.items():
            tau_post = cross_fitted_cate(X_post, T_post, Y_post, EstClass, n_splits=5)
            print(f"  {name} — Mean CATE: {tau_post.mean():+.4f}, "
                  f"Std: {tau_post.std():.4f}, "
                  f"% favoring AFB: {100*(tau_post > 0).mean():.1f}%")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()