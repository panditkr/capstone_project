"""
Part 3 - Ensembles, Tuning, and Full ML Pipeline
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_STATE = 42
FIG_DIR = 'figures'
LOG_PATH = 'logs/part3_log.txt'
log_lines = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log_lines.append(s)

art = joblib.load('part2_artifacts.pkl')
X_train_scaled = art['X_train_scaled']; X_test_scaled = art['X_test_scaled']
X_train = art['X_train']; X_test = art['X_test']
y_clf_train = art['y_clf_train']; y_clf_test = art['y_clf_test']
logreg_baseline = art['logreg_baseline']
feature_names = art['feature_names']

# ---------------------------------------------------------------
# Task 1: Decision Tree baseline (unconstrained)
# ---------------------------------------------------------------
log("="*70); log("TASK 1: UNCONSTRAINED DECISION TREE")
dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_full.fit(X_train_scaled, y_clf_train)
train_acc_full = accuracy_score(y_clf_train, dt_full.predict(X_train_scaled))
test_acc_full = accuracy_score(y_clf_test, dt_full.predict(X_test_scaled))
log(f"Train accuracy: {train_acc_full:.4f}  Test accuracy: {test_acc_full:.4f}  "
    f"Gap: {train_acc_full - test_acc_full:.4f}")

# ---------------------------------------------------------------
# Task 2: Controlled Decision Tree
# ---------------------------------------------------------------
log("="*70); log("TASK 2: CONTROLLED DECISION TREE (max_depth=5, min_samples_split=20)")
dt_ctrl = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE)
dt_ctrl.fit(X_train_scaled, y_clf_train)
train_acc_ctrl = accuracy_score(y_clf_train, dt_ctrl.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, dt_ctrl.predict(X_test_scaled))
log(f"Train accuracy: {train_acc_ctrl:.4f}  Test accuracy: {test_acc_ctrl:.4f}  "
    f"Gap: {train_acc_ctrl - test_acc_ctrl:.4f}")

# ---------------------------------------------------------------
# Task 3: Gini vs Entropy
# ---------------------------------------------------------------
log("="*70); log("TASK 3: GINI VS ENTROPY (max_depth=5)")
dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=RANDOM_STATE)
dt_gini.fit(X_train_scaled, y_clf_train)
acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=RANDOM_STATE)
dt_entropy.fit(X_train_scaled, y_clf_train)
acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))
log(f"Gini test accuracy: {acc_gini:.4f}")
log(f"Entropy test accuracy: {acc_entropy:.4f}")

# ---------------------------------------------------------------
# Task 4: Random Forest
# ---------------------------------------------------------------
log("="*70); log("TASK 4: RANDOM FOREST (n_estimators=100, max_depth=10)")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
log(f"Train accuracy: {rf_train_acc:.4f}  Test accuracy: {rf_test_acc:.4f}  AUC: {rf_auc:.4f}")

importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
top5_features = importances.head(5)
log("\nTop 5 features by importance:\n", top5_features.to_string())

# ---------------------------------------------------------------
# Task 4a: Gradient Boosting
# ---------------------------------------------------------------
log("="*70); log("TASK 4a: GRADIENT BOOSTING")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                 random_state=RANDOM_STATE)
gb.fit(X_train_scaled, y_clf_train)
gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
log(f"Train accuracy: {gb_train_acc:.4f}  Test accuracy: {gb_test_acc:.4f}  AUC: {gb_auc:.4f}")

# ---------------------------------------------------------------
# Task 4b: Feature ablation study
# ---------------------------------------------------------------
log("="*70); log("TASK 4b: FEATURE ABLATION (lowest 5 importance features removed)")
lowest5 = importances.tail(5).index.tolist()
log("Lowest 5 importance features:", lowest5)

X_train_reduced = X_train_scaled.drop(columns=lowest5)
X_test_reduced = X_test_scaled.drop(columns=lowest5)
rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf_reduced.fit(X_train_reduced, y_clf_train)
auc_full_model = rf_auc
auc_reduced_model = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
log(f"Full model AUC: {auc_full_model:.5f}")
log(f"Reduced model AUC (5 lowest-importance features removed): {auc_reduced_model:.5f}")
log(f"Difference: {auc_full_model - auc_reduced_model:.5f}")

# ---------------------------------------------------------------
# Task 5: Cross-validated comparison
# ---------------------------------------------------------------
log("="*70); log("TASK 5: 5-FOLD CROSS-VALIDATED AUC COMPARISON")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
models_for_cv = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
    'Decision Tree (depth=5)': DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
}
cv_results = {}
for name, model in models_for_cv.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=skf, scoring='roc_auc')
    cv_results[name] = (scores.mean(), scores.std())
    log(f"{name}: mean AUC = {scores.mean():.4f}  std = {scores.std():.4f}")

# ---------------------------------------------------------------
# Task 6: GridSearchCV on a Pipeline
# ---------------------------------------------------------------
log("="*70); log("TASK 6: GRIDSEARCHCV ON RANDOM FOREST PIPELINE")
pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          RandomForestClassifier(random_state=RANDOM_STATE))
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5],
}
grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_clf_train)  # unscaled -- pipeline handles scaling
log("Best params:", grid.best_params_)
log("Best CV score (AUC):", grid.best_score_)
n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
log(f"Total configurations evaluated: {n_configs} grid points x 5 folds = {n_configs*5} fits")

best_pipeline = grid.best_estimator_
test_auc_best = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
log("Best pipeline test-set AUC:", test_auc_best)

# ---------------------------------------------------------------
# Task 7: Manual learning curve
# ---------------------------------------------------------------
log("="*70); log("TASK 7: MANUAL LEARNING CURVE")
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
lc_rows = []
for f in fractions:
    n_rows = int(f * len(X_train))
    X_sub = X_train.iloc[:n_rows]
    y_sub = y_clf_train.iloc[:n_rows]
    pipe_f = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                            RandomForestClassifier(random_state=RANDOM_STATE,
                                                    **{k.split('__')[1]: v for k, v in grid.best_params_.items()}))
    pipe_f.fit(X_sub, y_sub)
    train_auc = roc_auc_score(y_sub, pipe_f.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, pipe_f.predict_proba(X_test)[:, 1])
    lc_rows.append([f, n_rows, train_auc, test_auc])
    log(f"fraction={f}  n={n_rows}  train_auc={train_auc:.4f}  test_auc={test_auc:.4f}")
lc_table = pd.DataFrame(lc_rows, columns=['Training fraction', 'n_rows', 'Training AUC', 'Test AUC'])

plt.figure(figsize=(7, 5))
plt.plot(lc_table['Training fraction'], lc_table['Training AUC'], marker='o', label='Training AUC')
plt.plot(lc_table['Training fraction'], lc_table['Test AUC'], marker='o', label='Test AUC')
plt.title('Learning Curve - Tuned Random Forest Pipeline')
plt.xlabel('Training data fraction')
plt.ylabel('ROC-AUC')
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/08_learning_curve.png', dpi=130)
plt.close()

# ---------------------------------------------------------------
# Task 8: Serialize best model
# ---------------------------------------------------------------
log("="*70); log("TASK 8: SERIALIZE BEST MODEL")
joblib.dump(best_pipeline, 'best_model.pkl')
log("Saved best_model.pkl")

# Reload + predict demo on two hand-crafted rows
reloaded = joblib.load('best_model.pkl')
demo_rows = X_test.iloc[:2]
demo_preds = reloaded.predict(demo_rows)
log("Reload-and-predict demo on 2 hand-crafted rows -> predictions:", demo_preds)

# ---------------------------------------------------------------
# Task 9: Summary comparison table
# ---------------------------------------------------------------
log("="*70); log("TASK 9: SUMMARY COMPARISON TABLE")
summary_rows = []
for name, (m, s) in cv_results.items():
    summary_rows.append([name, m, s])
summary_rows.append(['Tuned Random Forest (GridSearchCV)', grid.best_score_, np.nan])
summary_df = pd.DataFrame(summary_rows, columns=['Model', 'CV Mean AUC', 'CV Std AUC'])
test_auc_map = {
    'Logistic Regression': roc_auc_score(y_clf_test, logreg_baseline.predict_proba(X_test_scaled)[:, 1]),
    'Decision Tree (depth=5)': roc_auc_score(y_clf_test, dt_ctrl.predict_proba(X_test_scaled)[:, 1]),
    'Random Forest': rf_auc,
    'Gradient Boosting': gb_auc,
    'Tuned Random Forest (GridSearchCV)': test_auc_best,
}
summary_df['Test AUC'] = summary_df['Model'].map(test_auc_map)
log(summary_df.to_string(index=False))

with open(LOG_PATH, 'w') as f:
    f.write('\n'.join(log_lines))

# Save key numbers for README building
joblib.dump({
    'train_acc_full': train_acc_full, 'test_acc_full': test_acc_full,
    'train_acc_ctrl': train_acc_ctrl, 'test_acc_ctrl': test_acc_ctrl,
    'acc_gini': acc_gini, 'acc_entropy': acc_entropy,
    'rf_train_acc': rf_train_acc, 'rf_test_acc': rf_test_acc, 'rf_auc': rf_auc,
    'top5_features': top5_features, 'gb_train_acc': gb_train_acc,
    'gb_test_acc': gb_test_acc, 'gb_auc': gb_auc,
    'lowest5': lowest5, 'auc_full_model': auc_full_model, 'auc_reduced_model': auc_reduced_model,
    'cv_results': cv_results, 'best_params': grid.best_params_, 'best_score': grid.best_score_,
    'test_auc_best': test_auc_best, 'lc_table': lc_table, 'summary_df': summary_df,
}, 'part3_summary.pkl')

print("\n\nDONE. Log saved to", LOG_PATH)
