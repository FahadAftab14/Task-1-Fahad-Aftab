import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# STEP 0: LOAD DATASET

df = pd.read_excel('Dataset for Data Analytics.xlsx')

print("STEP 0: DATASET OVERVIEW\n")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")


# PHASE 1: INPUT — SECURING FIDELITY
# Missing Value Analysis & Imputation

print("PHASE 1: MISSING VALUE ANALYSIS\n")

missing_count = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing_count, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]
print(f"\nMissing Data Summary:\n{missing_df}")

print("Imputing CouponCode (25.75% missing)\n")
print("Strategy: Group-wise mode imputation by OrderStatus\n")

# Group-wise mode imputation
def group_mode_impute(df, col, group_col):
    df = df.copy()
    for group in df[group_col].unique():
        mask_group = df[group_col] == group
        mode_val = df.loc[mask_group & df[col].notna(), col].mode()
        if not mode_val.empty:
            df.loc[mask_group & df[col].isna(), col] = mode_val[0]
    # Fill any remaining NaN with overall mode
    overall_mode = df[col].mode()[0]
    df[col].fillna(overall_mode, inplace=True)
    return df

df = group_mode_impute(df, 'CouponCode', 'OrderStatus')

print(f"Missing after imputation: {df['CouponCode'].isnull().sum()}\n")
print(f"CouponCode value counts:\n{df['CouponCode'].value_counts()}")

# Verify no missing values remain
print(f"\nTotal missing values remaining: {df.isnull().sum().sum()}")

# OUTLIER DETECTION & NEUTRALIZATION (IQR Method)

print("\nPHASE 1: OUTLIER DETECTION & NEUTRALIZATION (IQR)\n")

numeric_cols = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return lower, upper, len(outliers)

print(f"\n{'Column':<15} {'Lower Bound':<15} {'Upper Bound':<15} {'Outliers Found'}")
print("-" * 60)
bounds = {}
for col in numeric_cols:
    lower, upper, n_out = detect_outliers_iqr(df, col)
    bounds[col] = (lower, upper)
    print(f"{col:<15} {lower:<15.2f} {upper:<15.2f} {n_out}")

# Winsorization
print("\nApplying Winsorization (numpy.clip)")
df_clean = df.copy()
for col in numeric_cols:
    lower, upper = bounds[col]
    before = df_clean[col].describe()
    df_clean[col] = np.clip(df_clean[col], lower, upper)
    after = df_clean[col].describe()
    print(f"\n{col}: clipped [{lower:.2f}, {upper:.2f}]")
    print(f"  Before => min:{before['min']:.2f}, max:{before['max']:.2f}, std:{before['std']:.2f}")
    print(f"  After  => min:{after['min']:.2f},  max:{after['max']:.2f},  std:{after['std']:.2f}")

# Also detect using Z-Score for reference
print("\nZ-Score Outlier Count (|z| > 3)")
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df_clean[col]))
    print(f"{col}: {(z_scores > 3).sum()} outliers (post-IQR winsorization)")


# PHASE 2: PROCESS — VECTORIZED COMPUTATION ENGINE

print("\nPHASE 2: FEATURE ENGINEERING (6 new features)")

# Feature 1: Revenue Per Item 
df_clean['RevenuePerItem'] = df_clean['TotalPrice'] / df_clean['ItemsInCart']
print("\nFeature 1: RevenuePerItem = TotalPrice / ItemsInCart")
print(df_clean['RevenuePerItem'].describe())

# Feature 2: Price Efficiency Ratio 
df_clean['PriceEfficiencyRatio'] = df_clean['UnitPrice'] / df_clean['TotalPrice']
print("\nFeature 2: PriceEfficiencyRatio = UnitPrice / TotalPrice")
print(df_clean['PriceEfficiencyRatio'].describe())

# Feature 3: Order Month (temporal feature from Date)
df_clean['OrderMonth'] = df_clean['Date'].dt.month
print("\nFeature 3: OrderMonth (1-12 extracted from Date)")
print(df_clean['OrderMonth'].value_counts().sort_index())

# Feature 4: Order Quarter 
df_clean['OrderQuarter'] = df_clean['Date'].dt.quarter
print("\nFeature 4: OrderQuarter (1-4 extracted from Date)")
print(df_clean['OrderQuarter'].value_counts().sort_index())

# Feature 5: HasCoupon (binary flag) 
discount_coupons = ['SAVE10', 'WINTER15']
df_clean['HasDiscountCoupon'] = df_clean['CouponCode'].isin(discount_coupons).astype(int)
print("\nFeature 5: HasDiscountCoupon (1 if SAVE10/WINTER15, else 0)")
print(df_clean['HasDiscountCoupon'].value_counts())

# Feature 6: CartUtilizationRate
df_clean['CartUtilizationRate'] = df_clean['Quantity'] / df_clean['ItemsInCart']
print("\nFeature 6: CartUtilizationRate = Quantity / ItemsInCart")
print(df_clean['CartUtilizationRate'].describe())

print(f"\n Total features: {len(df_clean.columns)} (original 14 + 6 engineered)")

# CATEGORICAL ENCODING (One-Hot Encoding)

print("\nPHASE 2: CATEGORICAL ENCODING (One-Hot Encoding)\n")

cat_cols_to_encode = ['Product', 'PaymentMethod', 'OrderStatus', 'ReferralSource', 'CouponCode']

print(f"Columns to OHE: {cat_cols_to_encode}")
for col in cat_cols_to_encode:
    print(f"  {col}: {df_clean[col].nunique()} unique values → {df_clean[col].unique()}")

df_encoded = pd.get_dummies(df_clean, columns=cat_cols_to_encode, drop_first=False, dtype=int)
print(f"\nShape after OHE: {df_encoded.shape}")
print(f"New OHE columns added: {df_encoded.shape[1] - df_clean.shape[1]}")

# COLLINEARITY ERADICATION

print("\nPHASE 2: MULTICOLLINEARITY CHECK\n")

numeric_for_corr = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice',
                    'RevenuePerItem', 'PriceEfficiencyRatio', 'CartUtilizationRate',
                    'OrderMonth', 'OrderQuarter', 'HasDiscountCoupon']

corr_matrix = df_clean[numeric_for_corr].corr().abs()

# Isolate upper triangle
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find pairs with correlation > 0.80
high_corr_pairs = [(col, row, upper_tri.loc[row, col])
                   for col in upper_tri.columns
                   for row in upper_tri.index
                   if upper_tri.loc[row, col] > 0.80]

print(f"\nHighly correlated pairs (|r| > 0.80):")
cols_to_drop = set()
if high_corr_pairs:
    # Compute correlations with TotalPrice ONCE from the original corr_matrix
    target_corr = df_clean[numeric_for_corr].corr()['TotalPrice']
    for col1, col2, val in high_corr_pairs:
        print(f"  {col1} ↔ {col2}: r = {val:.4f}")
        # Skip pair if one already queued for dropping
        if col1 in cols_to_drop or col2 in cols_to_drop:
            print(f" Skipped (one already marked for drop)")
            continue
        corr1 = abs(target_corr.get(col1, 0))
        corr2 = abs(target_corr.get(col2, 0))
        drop = col1 if corr1 < corr2 else col2
        keep = col2 if corr1 < corr2 else col1
        print(f" Dropping '{drop}' (r={min(corr1,corr2):.4f} with TotalPrice) | Keeping '{keep}' (r={max(corr1,corr2):.4f})")
        cols_to_drop.add(drop)
    # Drop all at once
    actual_drops = [c for c in cols_to_drop if c in df_clean.columns]
    if actual_drops:
        df_clean.drop(columns=actual_drops, inplace=True)
        print(f"\n  Dropped columns: {actual_drops}")
else:
    print("  No highly correlated pairs found. All features retained.")

# PHASE 3: OUTPUT — FINAL CLEAN DATASET SUMMARY

print("\nPHASE 3: FINAL DATASET SUMMARY")

print(f"\nFinal Shape: {df_clean.shape}")
print(f"Missing Values: {df_clean.isnull().sum().sum()}")
print(f"\nFinal Columns:\n{df_clean.columns.tolist()}")
print(f"\nDescriptive Statistics (numeric):\n{df_clean.describe().round(2)}")

# Save cleaned dataset
df_clean.to_csv('cleaned_dataset.csv', index=False)
print("\n Cleaned dataset saved to: cleaned_dataset.csv")

# VISUALIZATIONS

print("\nGENERATING VISUALIZATIONS")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('EDA & Feature Engineering — Dashboard', fontsize=16, fontweight='bold')

# 1. Missing data heatmap (original)
ax1 = axes[0, 0]
miss_data = pd.DataFrame({'Column': ['CouponCode'], 'Missing %': [25.75]})
ax1.barh(miss_data['Column'], miss_data['Missing %'], color='tomato')
ax1.axvline(x=5, color='green', linestyle='--', label='5% threshold')
ax1.axvline(x=20, color='orange', linestyle='--', label='20% threshold')
ax1.set_title('Missing Data % (Before Imputation)')
ax1.set_xlabel('Missing %')
ax1.legend(fontsize=7)

# 2. Distribution of TotalPrice (before vs after winsorization)
ax2 = axes[0, 1]
ax2.hist(df['TotalPrice'], bins=40, alpha=0.6, color='salmon', label='Before Winsor.')
ax2.hist(df_clean['TotalPrice'], bins=40, alpha=0.6, color='steelblue', label='After Winsor.')
ax2.set_title('TotalPrice Distribution: Before vs After Winsorization')
ax2.set_xlabel('TotalPrice')
ax2.legend()

# 3. Boxplots of numeric features (after cleaning)
ax3 = axes[0, 2]
df_clean[['Quantity', 'ItemsInCart']].boxplot(ax=ax3)
ax3.set_title('Boxplot: Quantity & ItemsInCart (Post-Cleaning)')

# 4. UnitPrice by Product
ax4 = axes[1, 0]
df_clean.groupby('Product')['UnitPrice'].median().sort_values().plot(kind='barh', ax=ax4, color='teal')
ax4.set_title('Median UnitPrice by Product')
ax4.set_xlabel('Median UnitPrice')

# 5. Order Status Distribution
ax5 = axes[1, 1]
df_clean['OrderStatus'].value_counts().plot(kind='pie', ax=ax5, autopct='%1.1f%%', startangle=90)
ax5.set_title('Order Status Distribution')
ax5.set_ylabel('')

# 6. Orders by Month (new feature)
ax6 = axes[1, 2]
df_clean['OrderMonth'].value_counts().sort_index().plot(kind='bar', ax=ax6, color='mediumpurple')
ax6.set_title('Feature 3: Orders by Month')
ax6.set_xlabel('Month')
ax6.set_ylabel('Order Count')

# 7. RevenuePerItem by Product (new feature)
ax7 = axes[2, 0]
df_clean.groupby('Product')['RevenuePerItem'].mean().sort_values().plot(kind='barh', ax=ax7, color='darkorange')
ax7.set_title('Feature 1: Avg RevenuePerItem by Product')

# 8. CartUtilizationRate distribution (new feature)
ax8 = axes[2, 1]
ax8.hist(df_clean['CartUtilizationRate'], bins=30, color='seagreen', edgecolor='white')
ax8.set_title('Feature 6: CartUtilizationRate Distribution')
ax8.set_xlabel('Quantity / ItemsInCart')

# 9. Correlation heatmap of final numeric features
ax9 = axes[2, 2]
final_numeric = ['Quantity', 'UnitPrice', 'TotalPrice', 'ItemsInCart',
                 'RevenuePerItem', 'CartUtilizationRate', 'HasDiscountCoupon']
corr_final = df_clean[[c for c in final_numeric if c in df_clean.columns]].corr()
sns.heatmap(corr_final, annot=True, fmt='.2f', cmap='coolwarm', ax=ax9,
            annot_kws={'size': 7}, linewidths=0.5)
ax9.set_title('Correlation Heatmap (Final Features)')
ax9.tick_params(axis='x', rotation=45, labelsize=7)
ax9.tick_params(axis='y', rotation=0, labelsize=7)

plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualizations saved to: eda_dashboard.png")