# Dubai Residential Transactions 2026 — End-to-End ML Project

**Dataset**: `dubai_residential_data_2026.csv` — 18,085 individual property transaction
records from the Dubai residential real-estate market in 2026, with 15 columns covering
the transaction (date, procedure, sale value), the unit (area, room type, sqm), and
location context (nearest metro/mall/landmark, project name). It is a realistic,
messy administrative dataset: several categorical columns have meaningful missingness,
one numeric column (`TRANS_VALUE`) is extremely right-skewed as is typical of price
data, and two columns (`size_category`, `value_band`) are pre-computed bins of other
columns that must be handled carefully to avoid leakage in Part 2 onward.

This README documents all four parts of the project. Code lives in the four `partN_*.py`
scripts; each runs top-to-bottom independently (Part 2 depends on Part 1's
`cleaned_data.csv`, Part 3 depends on Part 2's `part2_artifacts.pkl`, Part 4 depends on
Part 3's `best_model.pkl`). Full console output for each part is saved under `logs/`.

---

## Part 1 — Data Acquisition, Cleaning, and EDA

**Script**: `part1_eda.py` → produces `cleaned_data.csv`, `figures/01`–`06`, `logs/part1_log.txt`

### Null analysis
| Column | Null % |
|---|---|
| NEAREST_MALL_EN | 20.75% |
| PROJECT_EN | 22.49% |
| NEAREST_METRO_EN | 19.75% |
| NEAREST_LANDMARK_EN | 9.13% |
| ROOMS_EN | 2.00% |
| all numeric columns (TRANS_VALUE, ACTUAL_AREA, price_per_sqm) | 0.00% |

`NEAREST_MALL_EN` and `PROJECT_EN` exceed the 20% threshold. Because **no numeric column
has any missing values** in this dataset, the median-fill rule for numeric columns is a
no-op here — all missingness lives in categorical text fields. Missing categorical
values (below and above the 20% threshold) were filled with the explicit placeholder
category `'Unknown'` rather than dropped, since a missing "nearest landmark" is itself
informative (it usually means the property is in a newer or more remote development)
and dropping ~20% of rows would meaningfully shrink and bias the dataset. We chose
**median over mean** as the general numeric-fill rule because `TRANS_VALUE`,
`ACTUAL_AREA`, and `price_per_sqm` are all heavily right-skewed (see below) — the mean
of a skewed distribution is pulled toward the extreme high tail, so it is not a
representative "typical" value, while the median is robust to that skew.

### Duplicates
`df.duplicated().sum()` = **0**. No rows were removed, so null percentages were
unaffected by deduplication (verified in the log).

### Dtype correction
- `INSTANCE_DATE` was stored as a generic string and was converted to `datetime64` with
  `pd.to_datetime()`.
- Nine repetitive string columns (`PROCEDURE_EN`, `IS_FREE_HOLD_EN`, `AREA_EN`,
  `PROP_SB_TYPE_EN`, `ROOMS_EN`, `NEAREST_METRO_EN`, `NEAREST_MALL_EN`,
  `NEAREST_LANDMARK_EN`, `size_category`, `value_band`) were converted to `category` dtype.
- Memory usage: **13,436,266 bytes → 1,921,524 bytes (85.7% smaller)**.

### Descriptive statistics & skewness
| Column | skew |
|---|---|
| price_per_sqm | **68.95** (most skewed) |
| ACTUAL_AREA | 16.54 |
| TRANS_VALUE | 11.31 |

All three numeric columns are **strongly positively skewed** — a long right tail of a
small number of very expensive / very large / very high-rate properties (e.g. the max
`TRANS_VALUE` is AED 100,000,000 against a median of AED 1,190,000). For a positively
skewed column, the mean is pulled well above the typical value by these extreme highs;
imputing missing values with the mean would systematically overstate the "typical"
transaction, which is why median imputation is the correct default here (see the Part 1
justification above and the Task 8a comparison below).

### IQR outlier analysis
| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers |
|---|---|---|---|---|---|---|
| TRANS_VALUE | 730,000 | 2,098,512 | 1,368,512 | −1,322,768 | 4,151,280 | rows with TRANS_VALUE above ~AED 4.15M or below the (non-binding) lower bound |
| ACTUAL_AREA | 54.77 | 115.53 | 60.76 | −36.37 | 206.67 | rows with area above ~207 sqm |

We chose **not to drop** these outliers. They correspond to genuine ultra-premium
transactions (Palm Jumeirah villas, penthouses, Burj Khalifa units) rather than data
entry errors — removing them would bias the model toward mid-market properties and make
it unable to price the high end at all. In Part 2/3 we instead rely on tree-based models
(Random Forest, Gradient Boosting), which are naturally robust to extreme values because
they split on rank/threshold rather than raw magnitude, and on Ridge regression, whose
L2 penalty limits how much any single extreme-leverage point can dominate the fitted
coefficients.

### Visualizations
1. **Line plot** (`01_line_price_per_sqm_over_time.png`) — mean price per sqm by
   transaction date across 2026; shows day-to-day volatility typical of transaction-level
   averages with no strong long-run trend over the observed window.
2. **Bar chart** (`02_bar_meanvalue_by_sizecat.png`) — mean `TRANS_VALUE` rises
   monotonically from Compact → Mid-Size → Spacious → Premium, confirming unit size is a
   strong price driver.
3. **Histogram** (`03_hist_most_skewed.png`) — `price_per_sqm` distribution is a tall
   spike near AED 10,000–20,000/sqm with a long thin tail out past AED 2,000,000/sqm
   (a handful of ultra-luxury micro-transactions), visually confirming the skew=68.95
   statistic.
4. **Scatter plot** (`04_scatter_area_vs_value.png`) — `ACTUAL_AREA` vs `TRANS_VALUE`
   shows a clear **positive, moderately strong** relationship (Pearson r ≈ 0.70): larger
   units cost more, as expected, with increasing spread (heteroscedasticity) at larger
   sizes since luxury finishes/location add extra price variance on top of size alone.
5. **Box plot** (`05_box_value_by_band.png`) — median and spread of `TRANS_VALUE` step
   up cleanly from Entry → Mid-Market → High-End → Ultra-Premium, with the IQR box
   widening at higher bands, showing both higher typical price and higher price
   variability among more expensive properties.
6. **Correlation heat map** (`06_corr_heatmap.png`) — computed on `TRANS_VALUE`,
   `ACTUAL_AREA`, `price_per_sqm`.

### Correlation heat map interpretation
The highest correlation pair is **TRANS_VALUE ↔ ACTUAL_AREA (r ≈ 0.70)**. This is very
plausibly a direct causal relationship (more floor area requires more materials, more
land value, and more finishing cost, so it mechanically raises price) rather than pure
confounding. That said, a plausible **alternative/confounding explanation** is
*location/building quality*: prime areas (e.g. Palm Jumeirah, Downtown Dubai) tend to
build larger units **and** charge a location premium per sqm, so part of the
area–value correlation could be inflated by a third variable — the neighborhood/project
prestige — rather than square footage alone. This is exactly why the Part 2 models
include location dummies (`AREA_EN`) alongside `ACTUAL_AREA`, so the two effects can be
partially disentangled.

### Task 8a — Mean vs median for the two most-skewed columns
| Column | Mean | Median | Skew | Chosen statistic |
|---|---|---|---|---|
| price_per_sqm | 18,644.98 | **15,795.17** | +68.95 | Median |
| ACTUAL_AREA | 94.38 | **78.11** | +16.54 | Median |

Both columns are strongly **positively** skewed, so their means are pulled upward by a
small number of extreme high values (ultra-premium price/sqm, very large units). The
median is therefore the more representative central tendency for both, and is the
statistic used for imputation (`isnull().sum()` confirms 0 remaining nulls in both after
`fillna()`).

### Task 8b — Spearman vs Pearson
With only three numeric columns, all three possible pairs are reported (there is no
larger set to select a "top 3" from):

| Pair | Pearson | Spearman | \|Δ\| |
|---|---|---|---|
| TRANS_VALUE ↔ price_per_sqm | 0.526 | 0.678 | **0.152** (largest) |
| TRANS_VALUE ↔ ACTUAL_AREA | 0.699 | 0.758 | 0.058 |
| ACTUAL_AREA ↔ price_per_sqm | 0.062 | 0.086 | 0.025 |

For all three pairs, |Spearman| > |Pearson|, meaning each relationship is **monotonic
but not perfectly linear** — most notably `TRANS_VALUE` vs `price_per_sqm`, where the
Pearson correlation understates how consistently the two variables move together in
rank order (extreme high-value/high-rate transactions pull the linear correlation down
relative to the rank-based one). Because all three relationships are more monotonic
than linear, **Spearman correlation is the more informative measure for feature
selection guidance going into Part 2** — it won't be misled by the handful of extreme
outliers that compress the Pearson estimate.

### Task 8c — Grouped aggregation
Grouped `TRANS_VALUE` by `AREA_EN` (areas with ≥30 transactions, for stable statistics):
- **Highest mean**: `BLUEWATERS` (≈ AED 11.03M)
- **Highest std**: `BLUEWATERS` (≈ AED 12.33M) — the same area has both the highest
  average price and the widest price spread.
- **Ratio of highest to lowest group mean**: ≈ **21.2x** (Bluewaters vs. International
  City Ph 1)

A ratio this large (>20x) indicates `AREA_EN` carries **strong predictive signal** for
price — which area a unit is in explains a huge amount of the price variation on its
own. However, Bluewaters' very high within-group standard deviation (std ≈ mean) is a
real concern for a model that relies on the area feature alone: it means knowing "this
is a Bluewaters unit" narrows down the price only loosely, and the model still needs
other features (size, room count) to price individual units within that area reliably.

Full cleaned data is saved to **`cleaned_data.csv`** (18,085 rows × 15 columns after
cleaning/imputation).

---

## Part 2 — Supervised ML: Regression + Classification

**Script**: `part2_models.py` → produces `part2_artifacts.pkl`, `figures/07`, `logs/part2_log.txt`

### Label definitions
- **`y_reg`** = `TRANS_VALUE` (continuous, AED) — the transaction sale price.
- **`y_clf`** = `1` if `TRANS_VALUE` > its median (AED 1,190,000), else `0` — an
  above/below-median binary price-tier label, split almost exactly 50/50
  (49.7% / 50.3%).

### Feature set and leakage exclusions
`X` = `PROCEDURE_EN`, `IS_FREE_HOLD_EN`, `AREA_EN`, `PROP_SB_TYPE_EN`, `ACTUAL_AREA`,
`ROOMS_EN`, `NEAREST_METRO_EN`, `NEAREST_MALL_EN`, `NEAREST_LANDMARK_EN`, and an
engineered `TRANS_MONTH` (extracted from `INSTANCE_DATE`).

Five columns were deliberately **excluded** from `X`, beyond the target itself:
- `price_per_sqm` — mathematically `TRANS_VALUE / ACTUAL_AREA`, i.e. a near-perfect
  algebraic encoding of the regression target. Including it would be severe leakage.
- `size_category` — a direct bin of `ACTUAL_AREA` (verified empirically in the EDA).
- `value_band` — a direct bin of `TRANS_VALUE` (verified empirically) — direct label
  leakage for both the regression and classification targets.
- `PROJECT_EN` — 1,120 unique values with 22.5% missingness; too sparse/high-cardinality
  for one-hot encoding without target encoding, which was out of scope here.
- `INSTANCE_DATE` — replaced by the derived `TRANS_MONTH` feature instead of being fed
  in raw.

### Encoding
- **`IS_FREE_HOLD_EN`** → label-encoded 0/1 (Non Free Hold / Free Hold) — genuinely
  binary, so ordering is trivial and lossless.
- **`ROOMS_EN`** → ordinal-encoded, ordered by increasing median `ACTUAL_AREA` per
  category in the training data (Studio → 1 B/R → Unknown → Shop → Office → 2 B/R → 3
  B/R → 4 B/R → PENTHOUSE → 5 B/R → 6 B/R). This dataset's "room type" field mixes
  residential unit sizes with commercial types (Office, Shop), so a naive Studio<1BR<2BR…
  ordering wouldn't place Office/Shop correctly; ordering by typical unit size gives a
  defensible, data-driven monotonic scale instead.
- **`PROCEDURE_EN`, `AREA_EN`, `PROP_SB_TYPE_EN`, `NEAREST_METRO_EN`, `NEAREST_MALL_EN`,
  `NEAREST_LANDMARK_EN`** → one-hot encoded with `drop_first=True`. These have no
  natural order (e.g. area names), so label-encoding them as integers 0..N would imply a
  false numeric distance/ranking between categories that the model would wrongly treat
  as meaningful magnitude; one-hot encoding avoids that false ordinal relationship.

Final `X` shape after encoding: **18,085 rows × 171 columns**.

### Leak-free split & scaling
`train_test_split(X, y, test_size=0.2, random_state=42)` → train 14,468 / test 3,617.
`StandardScaler` was **fit only on `X_train`**, then used to `transform()` both
`X_train` and `X_test`. Fitting the scaler on the full dataset (train+test) would be
**data leakage**: the mean/std used to standardize every feature would then encode
information about the test set's distribution into the "training" process, letting the
model benefit indirectly from statistics it should never see before evaluation — this
inflates reported test performance relative to what the model would actually achieve in
production on truly unseen data.

### Regression — Linear Regression vs Ridge
| Model | MSE | R² |
|---|---|---|
| Linear Regression | 1,808,569,670,624 | 0.6632 |
| Ridge (alpha=1.0) | 1,808,337,492,197 | 0.6632 |

**Top 3 features by \|coefficient\|**: `ACTUAL_AREA` (+1,833,571), `AREA_EN_PALM
JUMEIRAH` (+622,989), `AREA_EN_BURJ KHALIFA` (+484,693).

A large **positive** coefficient (e.g. `ACTUAL_AREA`) means a one-standard-deviation
increase in that scaled feature is associated with that many AED **more** predicted
transaction value, holding other features fixed. A large **negative** coefficient would
mean the opposite — that feature being present/higher is associated with a lower
predicted price (none of the top-3 here are negative, since size and prestige location
both push price up).

Ridge produces almost identical MSE/R² here because the feature set, while wide (171
one-hot columns), doesn't have severe multicollinearity that plain OLS struggles with on
14k+ training rows — `alpha` controls the strength of the L2 penalty added to the loss
(`alpha * sum(coef^2)`), shrinking all coefficients toward zero and spreading credit
across correlated dummy variables (e.g. neighboring areas) rather than letting OLS assign
large, unstable weights to a few of them; with `alpha=1.0` this shrinkage is mild enough
to barely move the fit on this data.

### Classification — Logistic Regression
Class balance was checked and found near-50/50 (minority class = 49.7%), so no
resampling was strictly required by the 35% rule; `class_weight='balanced'` was used
anyway as the documented imbalance-handling technique (SMOTE via `imblearn` was not
available in this offline sandbox — see Environment Notes below).

- **Confusion matrix**: `[[1713, 148], [177, 1579]]`
- **Accuracy / Precision / Recall / F1** (both classes): **0.91** across the board
- **AUC = 0.9673**

**Precision** = TP / (TP + FP). **Recall** = TP / (TP + FN).

For this specific task — flagging whether a transaction is above the median price
band — false negatives (missing an actually-expensive property) and false positives
(flagging a mid-market property as premium) are roughly equally costly in a general
pricing-support tool, so **F1 / balanced accuracy** is a reasonable default target here
rather than skewing hard toward precision or recall; a downstream use case that
specifically feeds a luxury-marketing list would instead prioritize **precision**
(don't waste luxury-agent time on mid-market leads), while a use case flagging
"potentially under-valued, needs re-appraisal" listings would prioritize **recall**
(don't miss any expensive property).

**AUC = 0.9673** means the model, given a random above-median and a random
below-median transaction, ranks the above-median one higher **96.7% of the time** — a
very strong separation between the two classes.

### Threshold sensitivity (0.30–0.70)
| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.848 | 0.950 | 0.896 |
| 0.40 | 0.889 | 0.925 | **0.907** (max) |
| 0.50 | 0.914 | 0.899 | 0.907 |
| 0.60 | 0.935 | 0.864 | 0.898 |
| 0.70 | 0.944 | 0.823 | 0.880 |

The F1-maximizing threshold is **0.40** (tied essentially with 0.50). If the business
goal is to prioritize recall (catch as many above-median properties as possible, e.g.
for a luxury-lead pipeline that can tolerate some false positives), the threshold should
be **lowered** below 0.40 — the cost is more false positives (wasted follow-up on
mid-market leads). If the goal is precision (only act on very confident premium calls),
raise it toward 0.60–0.70 — the cost is missing some genuinely above-median properties
(lower recall).

### Regularization experiment (C=1.0 vs C=0.01)
| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.9143 | 0.8992 | 0.9673 |
| C=0.01 (strong L2) | 0.9060 | 0.9055 | 0.9654 |

`C` is the **inverse** of the regularization strength in scikit-learn's
`LogisticRegression` (`C = 1/λ`): a smaller `C` means a stronger L2 penalty, shrinking
coefficients more aggressively toward zero and trading a little training-set fit for a
simpler, lower-variance decision boundary. Here, reducing `C` to 0.01 very slightly
**worsened** AUC and precision, and very slightly **improved** recall — the baseline
model wasn't overfitting badly enough for the extra regularization to help.

### Bootstrap CI for the AUC difference
500 bootstrap resamples of the test set were drawn with `np.random.choice(...,
replace=True)`, and the AUC difference (C=1.0 minus C=0.01) was recomputed each time:

- **Mean AUC difference**: **+0.00177**
- **95% CI**: **[0.00006, 0.00363]**
- **Excludes zero**: **Yes**

Because the 95% interval excludes zero (barely — it sits just above it), the C=1.0
model's small AUC advantage over the heavily-regularized C=0.01 model appears to be a
consistent, if very small, effect across resamples of the test set rather than noise —
though practically speaking a ~0.0018 AUC difference is not a meaningful improvement for
most business decisions.

---

## Part 3 — Ensembles, Tuning, and Full Pipeline

**Script**: `part3_ensembles.py` → produces `best_model.pkl`, `figures/08`, `logs/part3_log.txt`

### Decision tree baseline vs controlled tree
| Model | Train acc | Test acc | Gap |
|---|---|---|---|
| Unconstrained (`max_depth=None`) | 0.9967 | 0.8947 | **0.102** |
| Controlled (`max_depth=5, min_samples_split=20`) | 0.8696 | 0.8772 | **−0.008** |

The unconstrained tree shows classic **overfitting**: near-perfect training accuracy but
a ~10-point drop on the test set. Decision trees are high-variance models because they
fit the training data greedily, split by split, choosing the locally-best split at each
node without ever revisiting or correcting an earlier decision — given enough depth they
can carve out a leaf for almost every training point, memorizing noise along with signal.

`max_depth` limits how many sequential splits any path through the tree can take,
capping how finely the tree can carve up the feature space (reduces variance, at some
cost in bias/expressiveness). `min_samples_split=20` prevents a node from splitting
further once it has fewer than 20 samples, stopping the tree from creating splits that
just chase noise in a handful of outlier rows. Together they nearly close the train/test
gap (even slightly *negative* here, i.e. test accuracy ≥ train accuracy, which can happen
with a small held-out test set and strong regularization).

### Gini vs Entropy (max_depth=5)
| Criterion | Test accuracy |
|---|---|
| Gini | 0.8767 |
| Entropy | 0.8701 |

**Gini impurity** = `1 − Σ pᵢ²`. **Entropy** = `−Σ pᵢ log₂(pᵢ)`. A node with **Gini = 0**
means all samples in that node belong to a single class — it is perfectly pure and needs
no further splitting. The two criteria gave very similar (Gini marginally better) results
here, which is typical — they usually produce comparable trees in practice.

### Random Forest
- Train accuracy: 0.8995, Test accuracy: 0.8933, **AUC: 0.9630**

**Top 5 features by importance**:
| Feature | Importance |
|---|---|
| ACTUAL_AREA | 0.311 |
| ROOMS_EN | 0.257 |
| NEAREST_LANDMARK_EN_Burj Al Arab | 0.040 |
| NEAREST_MALL_EN_Dubai Mall | 0.034 |
| NEAREST_METRO_EN_Creek Metro Station | 0.019 |

Random Forest feature importance is computed as the **average reduction in Gini
impurity** contributed by a feature across every split that uses it, averaged over all
trees in the forest — a measure of how much that feature helps separate classes
*wherever the forest chose to use it*, structural and non-linear. This differs from a
linear regression coefficient, which measures the marginal *linear* effect of a
one-unit (scaled) change in a feature holding all others fixed — coefficients can be
compared in sign and magnitude directly, while importances are always non-negative and
only meaningful in relative (not linear-effect) terms.

**Bagging concept**: each tree in the forest is trained on an independent **bootstrap
sample** (drawn with replacement) of the training rows, and at every split only a random
subset of √(number of features) candidate features is considered. This double
randomization decorrelates the individual trees — each one makes somewhat different
mistakes — so averaging their predictions cancels out much of the variance that plagues
any single deep decision tree, without adding much bias, since the trees are still each
allowed to grow fairly deep.

### Gradient Boosting (Task 4a)
Train accuracy: 0.9019, Test accuracy: 0.9013, **AUC: 0.9647** — slightly ahead of the
single Random Forest, consistent with boosting's sequential error-correction typically
edging out bagging on tabular data of this size.

### Feature ablation study (Task 4b)
Lowest-5-importance features (all niche `AREA_EN` dummies: `MINA RASHID`, `JUMEIRA BAY`,
`PEARL JUMEIRA`, `DUBAI HEALTHCARE CITY - PHASE 1`, `EMIRATE LIVING`) were removed and a
second Random Forest retrained:

- Full-model test AUC: **0.96300**
- Reduced-model test AUC: **0.96314**
- Difference: **−0.00014** (reduced model is marginally *higher*, i.e. no real drop)

These five features were **genuinely uninformative** — removing them did not hurt (and
trivially improved) AUC, which makes sense since they are near-zero-importance, rarely
populated one-hot dummies for tiny neighborhoods. This supports deploying the
**simpler, lower-dimensional model** in production: fewer input columns to validate,
monitor, and maintain, with no measurable accuracy cost, as long as this ablation
result is re-checked whenever the underlying area coverage changes materially.

### Cross-validated comparison (5-fold StratifiedKFold, ROC-AUC)
| Model | CV mean AUC | CV std AUC |
|---|---|---|
| Logistic Regression | 0.9656 | 0.0029 |
| Decision Tree (depth=5) | 0.9158 | 0.0054 |
| Random Forest | 0.9603 | 0.0030 |
| Gradient Boosting | 0.9615 | 0.0037 |

Cross-validation gives a more reliable estimate of generalization performance than one
train/test split because it evaluates the model on **five different held-out folds**
and reports the spread as well as the average — a single split's test score can be
lucky or unlucky depending on exactly which rows land in the test set, whereas the CV
standard deviation directly quantifies that split-to-split variability.

### GridSearchCV on the Random Forest pipeline
Pipeline: `SimpleImputer(median) → StandardScaler → RandomForestClassifier`.
Grid: `n_estimators ∈ {50,100,200} × max_depth ∈ {5,10,None} × min_samples_leaf ∈ {1,5}`
→ **18 configurations × 5 folds = 90 total model fits**.

- **Best params**: `n_estimators=200, max_depth=None, min_samples_leaf=1`
- **Best CV AUC**: 0.9701
- **Test-set AUC of best pipeline**: **0.9708**

Exhaustive Grid Search checks every combination in the grid, guaranteeing the best point
*within that grid*, but its cost grows multiplicatively with the number of
hyperparameters and values — 90 fits here, and it would explode quickly with a finer
grid. Randomized Search instead samples a fixed budget of random combinations from the
specified distributions, trading the guarantee of finding the grid's best combination
for the ability to explore a much larger/finer hyperparameter space at a fixed,
controllable compute cost — usually a better trade-off once there are more than 2–3
hyperparameters.

### Manual learning curve (Task 7)
| Training fraction | n rows | Training AUC | Test AUC |
|---|---|---|---|
| 0.2 | 2,893 | 1.0000 | 0.9649 |
| 0.4 | 5,787 | 1.0000 | 0.9680 |
| 0.6 | 8,680 | 1.0000 | 0.9691 |
| 0.8 | 11,574 | 1.0000 | 0.9699 |
| 1.0 | 14,468 | 1.0000 | 0.9708 |

(i) Training AUC stays essentially at 1.0 throughout — expected for an
unconstrained-depth Random Forest, which can memorize any training subset regardless of
size, so it never shows the "AUC decreases as data grows" pattern that would appear with
a lower-capacity model. (ii) Test AUC **does** increase steadily with more training data
(0.9649 → 0.9708), so more data is still helping. (iii) **Conclusion**: the model is
currently **data-limited, not capacity-limited** — test AUC is still climbing at 100% of
the available training data with no sign of a plateau, so collecting more transaction
records would likely improve performance further, more so than adding model complexity.

### Serialization
`best_pipeline` (the GridSearchCV winner) was saved to **`best_model.pkl`** with
`joblib.dump`. A reload-and-predict block (`joblib.load('best_model.pkl')` →
`.predict()` on two held-out rows) runs without errors, returning `[0, 1]`.

### Summary comparison & recommendation
| Model | CV mean AUC | CV std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.9656 | 0.0029 | 0.9673 |
| Decision Tree (depth=5) | 0.9158 | 0.0054 | 0.9242 |
| Random Forest | 0.9603 | 0.0030 | 0.9630 |
| Gradient Boosting | 0.9615 | 0.0037 | 0.9647 |
| **Tuned Random Forest (GridSearchCV)** | **0.9701** | — | **0.9708** |

**Recommendation**: deploy the **tuned Random Forest pipeline** (`best_model.pkl`). It
has the highest cross-validated and test AUC of all five models, it is packaged as a
single sklearn `Pipeline` (imputer + scaler + classifier) so it can be handed unscaled,
possibly-missing raw features directly with no separate preprocessing code to keep in
sync, and — per the ablation study — its performance is not fragile to a handful of
sparse location dummies, suggesting it will generalize reasonably as new areas/projects
appear in future data. Plain Logistic Regression is a very close second and remains a
good, fast, interpretable fallback if model transparency becomes a hard requirement.

---

## Part 4 — LLM-Powered Feature: Track C (Model Prediction Explanation Pipeline)

**Track chosen**: **(C) Model Prediction Explanation Pipeline**
**Script**: `part4_llm.py` → produces `logs/part4_log.txt`, `part4_summary.pkl`

### ⚠️ Environment note (read first)
This project was built and executed inside a **sandboxed container with no outbound
network access** (egress disabled). `call_llm()` is a complete, real implementation that
`requests.post()`s to a live OpenRouter-compatible chat-completions endpoint using an API
key from `os.environ['LLM_API_KEY']`, exactly per spec, and would work unmodified with
network access + a real key. Because no real HTTP call can succeed here, `call_llm()`
detects that failure and transparently falls back to a small, clearly-labeled
`offline_mock_llm()` rule-based stand-in so the **full pipeline** — prompting, JSON
parsing, schema validation, guardrails, temperature comparison — can still be
demonstrated end-to-end. Every fallback invocation is printed/logged
(`[call_llm] real API unavailable (...); using offline_mock_llm fallback`) so it is never
silently confused with a genuine model response. Similarly, the `jsonschema` package
could not be `pip install`-ed in this offline sandbox, so a small hand-rolled
`validate_schema()` function (required-field + type + enum checks, raising the same
`ValidationError` style) stands in for `jsonschema.validate()`.

### `call_llm` function
Implemented exactly per spec: builds the `{model, messages, temperature, max_tokens}`
JSON payload, sets `Authorization: Bearer <key>` + `Content-Type: application/json`
headers, POSTs, checks `status_code == 200`, and returns
`response.json()['choices'][0]['message']['content']` (or `None` on failure). Verified
with the sanity test prompt "Reply with only the word: hello" (log shows the fallback
firing and returning a valid response, since no network/key is present here).

### System prompt (verbatim)
```
You are a real-estate pricing model explainer. You will be given the raw feature values
of a property transaction, the trained model's predicted class (1 = predicted
transaction value above the dataset median, 0 = at or below median), and the model's
predicted probability for class 1. Respond with ONLY a single valid JSON object (no
markdown fences, no extra text) with exactly these fields: prediction_label (string),
confidence_level (one of: low, medium, high), top_reason (string), second_reason
(string), next_step (string). Base your reasoning only on the feature values provided;
do not invent facts.
```

### User prompt template (verbatim, with placeholders)
```
Feature values:
AREA_EN: {AREA_EN}
PROP_SB_TYPE_EN: {PROP_SB_TYPE_EN}
ACTUAL_AREA: {ACTUAL_AREA}
ROOMS_EN: {ROOMS_EN}
IS_FREE_HOLD_EN: {IS_FREE_HOLD_EN}
NEAREST_METRO_EN: {NEAREST_METRO_EN}
NEAREST_MALL_EN: {NEAREST_MALL_EN}
NEAREST_LANDMARK_EN: {NEAREST_LANDMARK_EN}
TRANS_MONTH: {TRANS_MONTH}

Predicted class: {pred_class}
Predicted probability of class 1 (above-median value): {pred_proba:.4f}

Return the JSON explanation now.
```

`temperature=0` was used for every "real" pipeline call because the task is a
**structured-data extraction/explanation** task where reproducibility matters more than
creative variety — at temperature near 0 the model always selects the highest-probability
next token, giving deterministic, repeatable JSON output for the same input, which is
what a downstream validation/monitoring system needs.

### Guardrail demonstration
`has_pii()` regex-checks for emails and 10-digit/dashed phone numbers before any LLM call.
- Input **with** an email (`"...jane.doe@example.com..."`) → **BLOCKED** ("Input blocked:
  PII detected.")
- Input **without** PII (`"Evaluate this 2-bedroom unit in Dubai Marina..."`) → **ALLOWED**,
  proceeds to `call_llm()`

### End-to-end demonstration (3 hand-crafted inputs)
`joblib.load('best_model.pkl')` → `encode_record(features)` (mirrors the Part 2
preprocessing: ordinal `ROOMS_EN`/`IS_FREE_HOLD_EN` + one-hot columns aligned to the
171-column training schema, filling absent dummies with 0) → `.predict()` /
`.predict_proba()` → structured LLM explanation → `json.loads()` → `validate_schema()`.

| Feature Input (abridged) | Predicted Class | Probability | Explanation JSON (abridged) | Validation |
|---|---|---|---|---|
| Palm Jumeirah, 3 B/R, 210 sqm | 1 | 0.9750 | `prediction_label: "above-median value"`, `confidence_level: "high"`, top_reason cites `ACTUAL_AREA=210.0` | **pass** |
| International City Ph 1, Studio, 45 sqm | 0 | 0.0000 | `prediction_label: "at-or-below-median value"`, `confidence_level: "high"` | **pass** |
| Business Bay, 2 B/R, 95.5 sqm | 1 | 0.9750 | `prediction_label: "above-median value"`, `confidence_level: "high"` | **pass** |

All three inputs produced valid JSON matching the 5-required-field schema (no failures
observed with the mock backend, since it is constructed to always emit the schema
exactly; with a real LLM, occasional failures — e.g. an added markdown code fence around
the JSON, or a missing field — are the typical failure pattern, which is why the
`try/except json.JSONDecodeError` + `try/except ValidationError` + fallback-to-null-dict
path exists and is exercised by the code even though it wasn't triggered in this run).

### Temperature A/B comparison (0.0 vs 0.7)
| Input | Output @ T=0 | Output @ T=0.7 | Key difference |
|---|---|---|---|
| Palm Jumeirah / 3 B/R | top_reason: "Property size is the dominant driver..." | top_reason: "Unit size stands out as the main factor..." | Same substantive conclusion, different wording of `top_reason` |
| International City Ph 1 / Studio | "Property size is the dominant driver..." | "Unit size stands out as the main factor..." | Wording variation only |
| Business Bay / 2 B/R | "Property size is the dominant driver..." | "The size of the unit weighed most heavily..." | Wording variation only |

At `temperature=0` the model (real or mock) always selects the single highest-probability
continuation, so re-running the same prompt returns byte-identical output — ideal for a
pipeline whose output gets machine-parsed and validated downstream. At `temperature=0.7`
the model samples from a wider slice of the output distribution, so *phrasing* varies
run to run even when the underlying facts and conclusion don't — useful for
customer-facing variety, but undesirable here where predictable, auditable output is the
goal, which is why `temperature=0` was the production choice for this pipeline.

---

## How to run
```bash
python3 part1_eda.py       # -> cleaned_data.csv, figures/01-06, logs/part1_log.txt
python3 part2_models.py    # -> part2_artifacts.pkl, figures/07, logs/part2_log.txt
python3 part3_ensembles.py # -> best_model.pkl, figures/08, logs/part3_log.txt
python3 part4_llm.py       # -> logs/part4_log.txt, part4_summary.pkl
```
To use a real LLM in Part 4: `export LLM_API_KEY=<your-openrouter-key>` and ensure the
environment has outbound network access — no code changes are required, `call_llm()`
will use the real endpoint automatically instead of the offline fallback.

## Repository contents
```
part1_eda.py            part2_models.py         part3_ensembles.py      part4_llm.py
cleaned_data.csv         best_model.pkl           README.md
figures/                 logs/
  01_line_price_per_sqm_over_time.png    part1_log.txt
  02_bar_meanvalue_by_sizecat.png        part2_log.txt
  03_hist_most_skewed.png                part3_log.txt
  04_scatter_area_vs_value.png           part4_log.txt
  05_box_value_by_band.png
  06_corr_heatmap.png
  07_roc_curve_logreg.png
  08_learning_curve.png
```
