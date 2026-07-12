"""
Part 1 - Data Acquisition, Cleaning, and Exploratory Analysis
Dubai Residential Transactions 2026
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = '/mnt/user-data/uploads/dubai_residential_data_2026.csv'
FIG_DIR = 'figures'
LOG_PATH = 'logs/part1_log.txt'

log_lines = []
def log(*args):
    s = ' '.join(str(a) for a in args)
    print(s)
    log_lines.append(s)

# ---------------------------------------------------------------
# Task 1: Load data
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
log("="*70); log("TASK 1: LOAD DATA")
log("First 5 rows:\n", df.head().to_string())
log("\nDtypes:\n", df.dtypes.to_string())
log("\nShape:", df.shape)

# ---------------------------------------------------------------
# Task 2: Null value analysis
# ---------------------------------------------------------------
log("="*70); log("TASK 2: NULL VALUE ANALYSIS")
null_counts = df.isnull().sum()
null_pct = (null_counts / df.shape[0]) * 100
null_table = pd.DataFrame({'null_count': null_counts, 'null_pct': null_pct.round(2)})
log(null_table.to_string())

high_null_cols = null_pct[null_pct > 20].index.tolist()
log("\nColumns exceeding 20% null rate:", high_null_cols)

numeric_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
log("\nNumeric columns (raw):", numeric_cols_raw)
# No numeric column has any nulls in this dataset (TRANS_VALUE, ACTUAL_AREA,
# price_per_sqm are all fully populated). All missingness lives in categorical
# text fields (ROOMS_EN, NEAREST_METRO_EN, NEAREST_MALL_EN, NEAREST_LANDMARK_EN,
# PROJECT_EN). We still apply the median-fill rule to numeric columns as
# instructed (a no-op here since there is nothing to fill), and additionally
# fill categorical columns below the 20% threshold with the placeholder
# category 'Unknown' so no information is silently dropped.
for col in numeric_cols_raw:
    if df[col].isnull().sum() > 0 and null_pct[col] <= 20:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        log(f"Filled numeric column '{col}' nulls with median={median_val}")

cat_cols_below_20 = [c for c in df.columns if c not in numeric_cols_raw
                     and df[c].isnull().sum() > 0 and c not in high_null_cols]
for col in cat_cols_below_20:
    df[col] = df[col].fillna('Unknown')
    log(f"Filled categorical column '{col}' nulls with 'Unknown' "
        f"(null rate {null_pct[col]:.2f}% < 20%)")

# Columns above 20% (NEAREST_MALL_EN, PROJECT_EN) are also filled with
# 'Unknown' rather than dropped, since PROJECT_EN / NEAREST_MALL_EN still
# carry usable signal for the rows that do have a value; they are flagged
# in the README and PROJECT_EN is additionally excluded from modeling in
# Part 2 due to very high cardinality (1120 unique values) on top of its
# high null rate.
for col in high_null_cols:
    df[col] = df[col].fillna('Unknown')
    log(f"Filled high-null categorical column '{col}' with 'Unknown' "
        f"(null rate {null_pct[col]:.2f}% > 20%, flagged in README)")

# ---------------------------------------------------------------
# Task 3: Duplicate detection & removal
# ---------------------------------------------------------------
log("="*70); log("TASK 3: DUPLICATES")
dup_count = df.duplicated().sum()
log("Duplicate rows found:", dup_count)
null_pct_before_dedup = (df.isnull().sum() / df.shape[0] * 100).round(2)
df = df.drop_duplicates()
log("Shape after dropping duplicates:", df.shape)
null_pct_after_dedup = (df.isnull().sum() / df.shape[0] * 100).round(2)
log("Null % changed after dedup?", not null_pct_before_dedup.equals(null_pct_after_dedup))
# (Nulls were already filled above, so both are all-zero; recorded for completeness.)

# ---------------------------------------------------------------
# Task 4: Data type correction
# ---------------------------------------------------------------
log("="*70); log("TASK 4: DTYPE CORRECTION")
mem_before = df.memory_usage(deep=True).sum()
log("Memory usage before conversion (bytes):", mem_before)

# INSTANCE_DATE is stored as a generic object/string -> should be datetime
df['INSTANCE_DATE'] = pd.to_datetime(df['INSTANCE_DATE'], errors='coerce')
log("Converted INSTANCE_DATE to datetime64. New dtype:", df['INSTANCE_DATE'].dtype)

# Repetitive string columns -> category dtype
cat_like_cols = ['PROCEDURE_EN', 'IS_FREE_HOLD_EN', 'AREA_EN', 'PROP_SB_TYPE_EN',
                  'ROOMS_EN', 'NEAREST_METRO_EN', 'NEAREST_MALL_EN',
                  'NEAREST_LANDMARK_EN', 'size_category', 'value_band']
for c in cat_like_cols:
    df[c] = df[c].astype('category')

mem_after = df.memory_usage(deep=True).sum()
log("Memory usage after conversion (bytes):", mem_after)
log(f"Memory reduction: {mem_before - mem_after} bytes "
    f"({(1 - mem_after/mem_before)*100:.1f}% smaller)")

# ---------------------------------------------------------------
# Task 5: Descriptive statistics & skewness
# ---------------------------------------------------------------
log("="*70); log("TASK 5: DESCRIBE + SKEWNESS")
numeric_cols = ['TRANS_VALUE', 'ACTUAL_AREA', 'price_per_sqm']
log(df[numeric_cols].describe().to_string())

skew_vals = {c: df[c].skew() for c in numeric_cols}
log("\nSkewness:", skew_vals)
most_skewed_col = max(skew_vals, key=lambda k: abs(skew_vals[k]))
log("Most skewed column:", most_skewed_col, "skew =", skew_vals[most_skewed_col])

# ---------------------------------------------------------------
# Task 6: Outlier detection with IQR
# ---------------------------------------------------------------
log("="*70); log("TASK 6: IQR OUTLIERS")
iqr_report = {}
for col in ['TRANS_VALUE', 'ACTUAL_AREA']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    iqr_report[col] = dict(Q1=Q1, Q3=Q3, IQR=IQR, lower=lower, upper=upper, n_outliers=n_outliers)
    log(f"{col}: Q1={Q1:.2f} Q3={Q3:.2f} IQR={IQR:.2f} "
        f"bounds=({lower:.2f}, {upper:.2f}) outliers={n_outliers} "
        f"({n_outliers/len(df)*100:.2f}% of rows)")

# ---------------------------------------------------------------
# Task 7: Visualizations
# ---------------------------------------------------------------
log("="*70); log("TASK 7: VISUALIZATIONS")

# 7a. Line plot: mean price_per_sqm over time
ts = df.sort_values('INSTANCE_DATE').groupby('INSTANCE_DATE')['price_per_sqm'].mean()
plt.figure(figsize=(10, 5))
plt.plot(ts.index, ts.values, color='#2c6e91')
plt.title('Mean Price per Sqm Over Time (Dubai Residential Transactions, 2026)')
plt.xlabel('Transaction Date')
plt.ylabel('Mean Price per Sqm (AED)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_line_price_per_sqm_over_time.png', dpi=130)
plt.close()

# 7b. Bar chart: mean TRANS_VALUE by size_category
plt.figure(figsize=(8, 5))
order = ['Compact', 'Mid-Size', 'Spacious', 'Premium']
means = df.groupby('size_category', observed=True)['TRANS_VALUE'].mean().reindex(order)
plt.bar(means.index, means.values, color='#3f8f5c')
plt.title('Mean Transaction Value by Size Category')
plt.xlabel('Size Category')
plt.ylabel('Mean Transaction Value (AED)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_bar_meanvalue_by_sizecat.png', dpi=130)
plt.close()

# 7c. Histogram of most skewed column
plt.figure(figsize=(8, 5))
sns.histplot(df[most_skewed_col], bins=20, color='#a04c4c')
plt.title(f'Distribution of {most_skewed_col} (skew={skew_vals[most_skewed_col]:.2f})')
plt.xlabel(most_skewed_col)
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_hist_most_skewed.png', dpi=130)
plt.close()

# 7d. Scatter plot: ACTUAL_AREA vs TRANS_VALUE
plt.figure(figsize=(8, 6))
sample = df.sample(min(3000, len(df)), random_state=42)
sns.scatterplot(data=sample, x='ACTUAL_AREA', y='TRANS_VALUE', alpha=0.4, s=15, color='#3d5a80')
plt.title('Actual Area vs Transaction Value')
plt.xlabel('Actual Area (sqm)')
plt.ylabel('Transaction Value (AED)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/04_scatter_area_vs_value.png', dpi=130)
plt.close()
pearson_area_value = df['ACTUAL_AREA'].corr(df['TRANS_VALUE'])
log("Pearson corr ACTUAL_AREA vs TRANS_VALUE:", pearson_area_value)

# 7e. Box plot: TRANS_VALUE by value_band
plt.figure(figsize=(9, 6))
band_order = ['Entry', 'Mid-Market', 'High-End', 'Ultra-Premium']
sns.boxplot(data=df, x='value_band', y='TRANS_VALUE', order=band_order, showfliers=False)
plt.title('Transaction Value Distribution by Value Band (outliers hidden for scale)')
plt.xlabel('Value Band')
plt.ylabel('Transaction Value (AED)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/05_box_value_by_band.png', dpi=130)
plt.close()

# 7f. Correlation heatmap
plt.figure(figsize=(6, 5))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap - Numeric Features')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/06_corr_heatmap.png', dpi=130)
plt.close()
log("\nPearson correlation matrix:\n", corr_matrix.to_string())

corr_pairs = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).abs().unstack().dropna()
top_pair = corr_pairs.idxmax()
log("Highest |correlation| pair:", top_pair, "value=", corr_pairs.max())

# ---------------------------------------------------------------
# Task 8a: Imputation strategy comparison (mean vs median for skewed cols)
# ---------------------------------------------------------------
log("="*70); log("TASK 8a: MEAN VS MEDIAN FOR TOP-2 SKEWED COLUMNS")
sorted_skew = sorted(skew_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)
top2_skewed = [c for c, _ in sorted_skew[:2]]
for c in top2_skewed:
    log(f"{c}: mean={df[c].mean():.2f}, median={df[c].median():.2f}, skew={skew_vals[c]:.3f}")
    # No nulls remain at this point (already handled in Task 2), but we still
    # run the required fillna() + isnull().sum() confirmation for the pipeline.
    fill_val = df[c].median() if abs(skew_vals[c]) > 0 else df[c].mean()
    df[c] = df[c].fillna(fill_val)
    log(f"  -> isnull().sum() after imputation: {df[c].isnull().sum()}")

# ---------------------------------------------------------------
# Task 8b: Spearman vs Pearson
# ---------------------------------------------------------------
log("="*70); log("TASK 8b: SPEARMAN VS PEARSON")
pearson_matrix = df[numeric_cols].corr(method='pearson')
spearman_matrix = df[numeric_cols].corr(method='spearman')
log("Pearson:\n", pearson_matrix.to_string())
log("\nSpearman:\n", spearman_matrix.to_string())

diff_matrix = (spearman_matrix - pearson_matrix).abs()
pairs = []
cols = numeric_cols
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        pairs.append((cols[i], cols[j], diff_matrix.iloc[i, j],
                      spearman_matrix.iloc[i, j], pearson_matrix.iloc[i, j]))
pairs_df = pd.DataFrame(pairs, columns=['col_a', 'col_b', 'abs_diff', 'spearman', 'pearson'])
pairs_df = pairs_df.sort_values('abs_diff', ascending=False)
log("\nDifference table (all pairs, since only 3 numeric columns exist):\n", pairs_df.to_string())

# ---------------------------------------------------------------
# Task 8c: Grouped aggregation
# ---------------------------------------------------------------
log("="*70); log("TASK 8c: GROUPED AGGREGATION")
group_agg = df.groupby('AREA_EN', observed=True)['TRANS_VALUE'].agg(['mean', 'std', 'count'])
group_agg = group_agg[group_agg['count'] >= 30]  # ignore tiny areas for stable stats
log(group_agg.sort_values('mean', ascending=False).to_string())

highest_mean_area = group_agg['mean'].idxmax()
highest_std_area = group_agg['std'].idxmax()
mean_ratio = group_agg['mean'].max() / group_agg['mean'].min()
log(f"\nHighest mean group: {highest_mean_area} ({group_agg['mean'].max():.0f})")
log(f"Highest std group: {highest_std_area} ({group_agg['std'].max():.0f})")
log(f"Ratio of highest to lowest group mean: {mean_ratio:.2f}")

# ---------------------------------------------------------------
# Save cleaned dataset
# ---------------------------------------------------------------
df.to_csv('cleaned_data.csv', index=False)
log("="*70)
log("Saved cleaned_data.csv with shape:", df.shape)

with open(LOG_PATH, 'w') as f:
    f.write('\n'.join(log_lines))

print("\n\nDONE. Log saved to", LOG_PATH)
