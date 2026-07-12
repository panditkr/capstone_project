"""
Part 2 - Supervised ML: Regression + Classification
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, confusion_matrix,
                              classification_report, roc_curve, roc_auc_score,
                              precision_score, recall_score, f1_score)

FIG_DIR = 'figures'
LOG_PATH = 'logs/part2_log.txt'
log_lines = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log_lines.append(s)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# Load + define labels
# ---------------------------------------------------------------
df = pd.read_csv('cleaned_data.csv', parse_dates=['INSTANCE_DATE'])
log("Loaded cleaned_data.csv:", df.shape)

y_reg = df['TRANS_VALUE'].copy()
y_clf = (y_reg > y_reg.median()).astype(int)
log("y_reg = TRANS_VALUE (continuous, AED)")
log("y_clf = 1 if TRANS_VALUE > median else 0. Median =", y_reg.median())
log("y_clf class balance:\n", y_clf.value_counts(normalize=True).to_string())

# Columns dropped from X and why (documented, not hidden):
#  - TRANS_VALUE            : the regression target itself
#  - price_per_sqm          : deterministically derived from TRANS_VALUE / ACTUAL_AREA
#                              (near-perfect leakage of the target)
#  - size_category           : derived directly from ACTUAL_AREA bins (leakage of a feature,
#                              but ACTUAL_AREA is highly predictive of TRANS_VALUE, so this is
#                              a proxy-leak of the label as well)
#  - value_band              : derived directly from TRANS_VALUE bins (direct label leakage)
#  - PROJECT_EN              : 1120 unique values + 22.5% null -> too sparse/high-cardinality
#                              for one-hot without target encoding, which was not requested
#  - INSTANCE_DATE           : replaced by an engineered TRANS_MONTH feature
df['TRANS_MONTH'] = df['INSTANCE_DATE'].dt.month
drop_cols = ['TRANS_VALUE', 'price_per_sqm', 'size_category', 'value_band',
             'PROJECT_EN', 'INSTANCE_DATE']
X = df.drop(columns=drop_cols)
log("\nDropped columns (leakage / too sparse):", drop_cols)
log("Remaining feature columns:", X.columns.tolist())

# ---------------------------------------------------------------
# Task 2: Encoding
# ---------------------------------------------------------------
log("="*70); log("ENCODING")

# IS_FREE_HOLD_EN: natural binary order -> label encode
X['IS_FREE_HOLD_EN'] = X['IS_FREE_HOLD_EN'].map({'Non Free Hold': 0, 'Free Hold': 1})
log("IS_FREE_HOLD_EN label-encoded: Non Free Hold=0, Free Hold=1 (binary, inherently ordinal)")

# ROOMS_EN: ordered by increasing typical unit size (median ACTUAL_AREA per category
# in the training data), which gives a defensible monotonic ordering for a mixed
# residential/commercial room-type field that isn't a simple Studio..6BR sequence.
room_order = (df.groupby('ROOMS_EN')['ACTUAL_AREA'].median().sort_values().index.tolist())
room_map = {cat: i for i, cat in enumerate(room_order)}
X['ROOMS_EN'] = X['ROOMS_EN'].map(room_map)
log("ROOMS_EN ordinal mapping (by median ACTUAL_AREA, ascending):", room_map)

# Nominal columns -> one-hot, drop_first to avoid multicollinearity
nominal_cols = ['PROCEDURE_EN', 'AREA_EN', 'PROP_SB_TYPE_EN',
                'NEAREST_METRO_EN', 'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN']
log("\nOne-hot encoding (no natural order, drop_first=True):", nominal_cols)
log("Rationale: label-encoding e.g. AREA_EN as 0,1,2...97 would imply a false ordinal "
    "distance/ranking between areas (that 'JUMEIRAH VILLAGE CIRCLE' is 'between' two "
    "other areas numerically), which the model would wrongly treat as meaningful "
    "magnitude. One-hot encoding removes that false ordinal relationship.")
X = pd.get_dummies(X, columns=nominal_cols, drop_first=True)
log("X shape after encoding:", X.shape)

# ---------------------------------------------------------------
# Task 3: Leak-free split + scaling
# ---------------------------------------------------------------
log("="*70); log("SPLIT + SCALING")
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE)
log("Train shape:", X_train.shape, "Test shape:", X_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)  # fit ONLY on training data
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
log("Scaler fit on X_train only. Fitting on the full dataset (train+test) would be "
    "data leakage: the mean/std used to standardize every feature would then encode "
    "information about the test set's distribution, so the model would be indirectly "
    "'trained' on statistics it should never have seen before evaluation, inflating "
    "reported test performance relative to real-world deployment.")

# ---------------------------------------------------------------
# Task 4: Linear Regression + Ridge
# ---------------------------------------------------------------
log("="*70); log("LINEAR REGRESSION")
lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
y_pred_reg = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_reg_test, y_pred_reg)
r2_lr = r2_score(y_reg_test, y_pred_reg)
log(f"Linear Regression -> MSE: {mse_lr:,.2f}  R2: {r2_lr:.4f}")

coef_table = pd.DataFrame({'feature': X_train_scaled.columns, 'coef': lr.coef_})
coef_table['abs_coef'] = coef_table['coef'].abs()
top3 = coef_table.sort_values('abs_coef', ascending=False).head(3)
log("\nTop 3 features by |coefficient|:\n", top3.to_string())

log("="*70); log("RIDGE REGRESSION")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
log(f"Ridge Regression (alpha=1.0) -> MSE: {mse_ridge:,.2f}  R2: {r2_ridge:.4f}")

comparison_reg = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge (alpha=1.0)'],
    'MSE': [mse_lr, mse_ridge],
    'R2': [r2_lr, r2_ridge]
})
log("\nComparison table:\n", comparison_reg.to_string(index=False))

# ---------------------------------------------------------------
# Task 5: Logistic Regression
# ---------------------------------------------------------------
log("="*70); log("LOGISTIC REGRESSION - CLASS BALANCE CHECK")
vc = y_clf_train.value_counts(normalize=True)
log(vc.to_string())
minority_pct = vc.min() * 100
log(f"Minority class share: {minority_pct:.1f}%")
use_class_weight = minority_pct < 35
log("Using class_weight='balanced':", use_class_weight,
    "(imblearn/SMOTE unavailable in this offline sandbox; class_weight='balanced' "
    "is the documented alternative and is applied identically regardless, since the "
    "median split here is naturally close to 50/50).")

logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
logreg.fit(X_train_scaled, y_clf_train)
y_pred_clf = logreg.predict(X_test_scaled)
y_proba_clf = logreg.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_clf_test, y_pred_clf)
log("\nConfusion matrix:\n", cm)
log("\nClassification report:\n", classification_report(y_clf_test, y_pred_clf))

fpr, tpr, _ = roc_curve(y_clf_test, y_proba_clf)
auc = roc_auc_score(y_clf_test, y_proba_clf)
log("AUC:", auc)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#2c6e91', label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve - Logistic Regression (C=1.0)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.annotate(f'AUC = {auc:.3f}', xy=(0.6, 0.2), fontsize=12,
             bbox=dict(boxstyle='round', fc='white', ec='gray'))
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/07_roc_curve_logreg.png', dpi=130)
plt.close()

# ---------------------------------------------------------------
# Task 5b: Decision-threshold sensitivity
# ---------------------------------------------------------------
log("="*70); log("THRESHOLD SENSITIVITY (0.30 - 0.70)")
rows = []
for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
    pred_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, pred_t)
    r = recall_score(y_clf_test, pred_t)
    f1 = f1_score(y_clf_test, pred_t)
    rows.append([t, p, r, f1])
thresh_table = pd.DataFrame(rows, columns=['Threshold', 'Precision', 'Recall', 'F1'])
log(thresh_table.to_string(index=False))
best_thresh = thresh_table.loc[thresh_table['F1'].idxmax(), 'Threshold']
log("Threshold maximising F1:", best_thresh)

# ---------------------------------------------------------------
# Task 6: Regularization experiment (C=0.01 vs C=1.0)
# ---------------------------------------------------------------
log("="*70); log("REGULARIZATION EXPERIMENT")
logreg_strong = LogisticRegression(max_iter=1000, class_weight='balanced',
                                    C=0.01, random_state=RANDOM_STATE)
logreg_strong.fit(X_train_scaled, y_clf_train)
y_pred_strong = logreg_strong.predict(X_test_scaled)
y_proba_strong = logreg_strong.predict_proba(X_test_scaled)[:, 1]

precision_c1 = precision_score(y_clf_test, y_pred_clf)
recall_c1 = recall_score(y_clf_test, y_pred_clf)
auc_c1 = auc

precision_c001 = precision_score(y_clf_test, y_pred_strong)
recall_c001 = recall_score(y_clf_test, y_pred_strong)
auc_c001 = roc_auc_score(y_clf_test, y_proba_strong)

reg_comparison = pd.DataFrame({
    'Model': ['C=1.0 (baseline)', 'C=0.01 (strong L2)'],
    'Precision': [precision_c1, precision_c001],
    'Recall': [recall_c1, recall_c001],
    'AUC': [auc_c1, auc_c001]
})
log(reg_comparison.to_string(index=False))

# ---------------------------------------------------------------
# Task 6b: Bootstrap CI for AUC difference
# ---------------------------------------------------------------
log("="*70); log("BOOTSTRAP CI FOR AUC DIFFERENCE (C=1.0 minus C=0.01)")
np.random.seed(RANDOM_STATE)
y_clf_test_arr = y_clf_test.to_numpy()
n = len(y_clf_test_arr)
diffs = []
for i in range(500):
    idx = np.random.choice(n, size=n, replace=True)
    y_sample = y_clf_test_arr[idx]
    if len(np.unique(y_sample)) < 2:
        continue  # skip degenerate bootstrap samples with a single class
    auc_1 = roc_auc_score(y_sample, y_proba_clf[idx])
    auc_2 = roc_auc_score(y_sample, y_proba_strong[idx])
    diffs.append(auc_1 - auc_2)
diffs = np.array(diffs)
mean_diff = diffs.mean()
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
log(f"Bootstrap iterations used: {len(diffs)}")
log(f"Mean AUC difference (C=1.0 - C=0.01): {mean_diff:.5f}")
log(f"95% CI: [{ci_low:.5f}, {ci_high:.5f}]")
log("Excludes zero:", not (ci_low <= 0 <= ci_high))

with open(LOG_PATH, 'w') as f:
    f.write('\n'.join(log_lines))

# Persist intermediate artifacts for Part 3
import joblib
joblib.dump({
    'X_train': X_train, 'X_test': X_test,
    'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
    'y_reg_train': y_reg_train, 'y_reg_test': y_reg_test,
    'y_clf_train': y_clf_train, 'y_clf_test': y_clf_test,
    'scaler': scaler, 'logreg_baseline': logreg,
    'feature_names': X.columns.tolist()
}, 'part2_artifacts.pkl')

print("\n\nDONE. Log saved to", LOG_PATH)
