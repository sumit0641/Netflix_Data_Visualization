import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

print("=" * 60)
print("STEP 1: DATA CLEANING AND HANDLING MISSING VALUES")
print("=" * 60)

# ============================================================================
# CREATE SYNTHETIC NETFLIX DATASET (Since we don't have the actual data)
# ============================================================================

print("\n📊 Creating Netflix Originals Dataset...")

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic Netflix Originals dataset
n_samples = 500

# Generate realistic Netflix data
titles = [f"Netflix Original {i}" for i in range(1, n_samples + 1)]
genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror', 'Romance', 'Sci-Fi', 
          'Documentary', 'Animation', 'Crime']
languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 
             'Japanese', 'Korean', 'Hindi', 'Mandarin']
content_types = ['Movie', 'TV Series', 'Limited Series', 'Documentary']

# Generate synthetic data with realistic patterns
data = {
    'Title': titles,
    'Genre': np.random.choice(genres, n_samples),
    'Release_Date': pd.date_range(start='2015-01-01', end='2024-12-31', periods=n_samples),
    'Runtime_Minutes': np.random.normal(120, 30, n_samples).astype(int),
    'Language': np.random.choice(languages, n_samples),
    'Content_Type': np.random.choice(content_types, n_samples),
    'Budget_Million_USD': np.random.lognormal(mean=2.5, sigma=0.8, size=n_samples),
    'IMDb_Rating': np.random.normal(6.8, 1.2, n_samples),
    'IMDb_Votes': np.random.lognormal(mean=8, sigma=1.5, size=n_samples).astype(int),
    'Netflix_Views_Million': np.random.lognormal(mean=2, sigma=1, size=n_samples),
    'Director_Experience_Years': np.random.exponential(scale=8, size=n_samples).astype(int),
    'Cast_Rating': np.random.normal(7.2, 1.5, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure ratings are within valid range
df['IMDb_Rating'] = df['IMDb_Rating'].clip(1, 10)
df['Cast_Rating'] = df['Cast_Rating'].clip(1, 10)

print(f"✅ Dataset created with {len(df)} Netflix Originals")
print(f"📋 Columns: {list(df.columns)}")

# ============================================================================
# INTRODUCE MISSING VALUES (Realistic Scenarios)
# ============================================================================

print("\n🔍 Introducing realistic missing values...")

# Introduce realistic missing values
missing_indices = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)

# IMDb Rating missing (some titles might not be rated yet)
df.loc[missing_indices[:25], 'IMDb_Rating'] = np.nan

# Budget missing (confidential information)
df.loc[missing_indices[25:50], 'Budget_Million_USD'] = np.nan

# Netflix Views missing (newer releases might not have view data)
df.loc[missing_indices[50:65], 'Netflix_Views_Million'] = np.nan

# Director Experience missing (new or unknown directors)
df.loc[missing_indices[65:75], 'Director_Experience_Years'] = np.nan

print(f"🎯 Missing values introduced across {len(missing_indices)} records")

# ============================================================================
# MISSING VALUE ANALYSIS
# ============================================================================

print("\n📈 MISSING VALUE ANALYSIS")
print("-" * 30)

# Check for missing values
missing_summary = df.isnull().sum()
missing_percentage = (missing_summary / len(df)) * 100

print("Missing Values Summary:")
print("+" + "-" * 40 + "+")
print("| {:<25} | {:>5} | {:>6} |".format("Column", "Count", "Percent"))
print("+" + "-" * 40 + "+")

for col in df.columns:
    count = missing_summary[col]
    percent = missing_percentage[col]
    if count > 0:
        print("| {:<25} | {:>5} | {:>5.1f}% |".format(col, count, percent))

print("+" + "-" * 40 + "+")

# Visualize missing data patterns
plt.figure(figsize=(14, 8))

# Missing data heatmap
plt.subplot(2, 2, 1)
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title('Missing Data Pattern Heatmap')

# Missing data bar chart
plt.subplot(2, 2, 2)
missing_cols = missing_summary[missing_summary > 0]
plt.bar(range(len(missing_cols)), missing_cols.values, color='coral')
plt.xticks(range(len(missing_cols)), missing_cols.index, rotation=45)
plt.title('Missing Values by Column')
plt.ylabel('Count of Missing Values')

# Missing data correlation
plt.subplot(2, 2, 3)
missing_matrix = df.isnull().astype(int)
missing_corr = missing_matrix.corr()
sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Missing Data Correlation')

# Data completeness over time
plt.subplot(2, 2, 4)
df['Release_Year'] = df['Release_Date'].dt.year
yearly_completeness = df.groupby('Release_Year').apply(
    lambda x: (1 - x.isnull().sum().sum() / (len(x) * len(x.columns))) * 100
)
plt.plot(yearly_completeness.index, yearly_completeness.values, marker='o')
plt.title('Data Completeness by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Completeness (%)')

plt.tight_layout()
plt.show()

# ============================================================================
# MISSING VALUE HANDLING STRATEGIES
# ============================================================================

print("\n🔧 MISSING VALUE HANDLING STRATEGIES")
print("-" * 40)

# Create a copy for cleaning
df_clean = df.copy()

# Strategy 1: IMDb Rating - Group-based median imputation
print("1. IMDb Rating - Strategic Imputation")
print("   Using median rating by Genre and Content Type")

missing_imdb_before = df_clean['IMDb_Rating'].isnull().sum()

for genre in df_clean['Genre'].unique():
    for content_type in df_clean['Content_Type'].unique():
        # Find records with missing IMDb rating for this genre-content combination
        mask = (df_clean['Genre'] == genre) & \
               (df_clean['Content_Type'] == content_type) & \
               df_clean['IMDb_Rating'].isnull()
        
        if mask.sum() > 0:
            # Calculate median for this group
            group_median = df_clean[
                (df_clean['Genre'] == genre) & 
                (df_clean['Content_Type'] == content_type)
            ]['IMDb_Rating'].median()
            
            # If no median available for this group, use overall median
            if pd.isna(group_median):
                group_median = df_clean['IMDb_Rating'].median()
            
            # Impute missing values
            df_clean.loc[mask, 'IMDb_Rating'] = group_median

missing_imdb_after = df_clean['IMDb_Rating'].isnull().sum()
print(f"   ✅ Reduced from {missing_imdb_before} to {missing_imdb_after} missing values")

# Strategy 2: Budget - Content Type based imputation
print("\n2. Budget - Content Type Based Imputation")
print("   Using median budget by Content Type")

missing_budget_before = df_clean['Budget_Million_USD'].isnull().sum()

for content_type in df_clean['Content_Type'].unique():
    mask = (df_clean['Content_Type'] == content_type) & df_clean['Budget_Million_USD'].isnull()
    
    if mask.sum() > 0:
        type_median = df_clean[df_clean['Content_Type'] == content_type]['Budget_Million_USD'].median()
        if pd.isna(type_median):
            type_median = df_clean['Budget_Million_USD'].median()
        df_clean.loc[mask, 'Budget_Million_USD'] = type_median

missing_budget_after = df_clean['Budget_Million_USD'].isnull().sum()
print(f"   ✅ Reduced from {missing_budget_before} to {missing_budget_after} missing values")

# Strategy 3: Netflix Views - Time-based imputation
print("\n3. Netflix Views - Time-based Imputation")
print("   Using median views by Release Year")

missing_views_before = df_clean['Netflix_Views_Million'].isnull().sum()

for year in df_clean['Release_Year'].unique():
    mask = (df_clean['Release_Year'] == year) & df_clean['Netflix_Views_Million'].isnull()
    
    if mask.sum() > 0:
        year_median = df_clean[df_clean['Release_Year'] == year]['Netflix_Views_Million'].median()
        if pd.isna(year_median):
            year_median = df_clean['Netflix_Views_Million'].median()
        df_clean.loc[mask, 'Netflix_Views_Million'] = year_median

missing_views_after = df_clean['Netflix_Views_Million'].isnull().sum()
print(f"   ✅ Reduced from {missing_views_before} to {missing_views_after} missing values")

# Strategy 4: Director Experience - Simple median imputation
print("\n4. Director Experience - Median Imputation")

missing_exp_before = df_clean['Director_Experience_Years'].isnull().sum()
median_experience = df_clean['Director_Experience_Years'].median()
df_clean['Director_Experience_Years'].fillna(median_experience, inplace=True)
missing_exp_after = df_clean['Director_Experience_Years'].isnull().sum()
print(f"   ✅ Reduced from {missing_exp_before} to {missing_exp_after} missing values")

# ============================================================================
# DATA CLEANING VALIDATION
# ============================================================================

print("\n✅ DATA CLEANING VALIDATION")
print("-" * 30)

# Final missing value check
final_missing = df_clean.isnull().sum()
total_missing = final_missing.sum()

print(f"Final Missing Values: {total_missing}")
if total_missing == 0:
    print("🎉 All missing values successfully handled!")
else:
    print("Remaining missing values by column:")
    for col, count in final_missing[final_missing > 0].items():
        print(f"   - {col}: {count}")

# Data type optimization
print("\n🔄 Optimizing Data Types...")
df_clean['Release_Date'] = pd.to_datetime(df_clean['Release_Date'])
df_clean['Runtime_Minutes'] = df_clean['Runtime_Minutes'].astype('int16')
df_clean['Director_Experience_Years'] = df_clean['Director_Experience_Years'].astype('int8')
df_clean['Release_Year'] = df_clean['Release_Year'].astype('int16')

# Remove any potential duplicates
initial_count = len(df_clean)
df_clean = df_clean.drop_duplicates()
final_count = len(df_clean)
duplicates_removed = initial_count - final_count

print(f"📊 Duplicates removed: {duplicates_removed}")

# Quality checks
print("\n🔍 Quality Assurance Checks:")
print(f"   - Total records: {len(df_clean)}")
print(f"   - Date range: {df_clean['Release_Date'].min().strftime('%Y-%m-%d')} to {df_clean['Release_Date'].max().strftime('%Y-%m-%d')}")
print(f"   - Rating range: {df_clean['IMDb_Rating'].min():.1f} to {df_clean['IMDb_Rating'].max():.1f}")
print(f"   - Complete records: {len(df_clean[df_clean.isnull().sum(axis=1) == 0])}")

# Before/After comparison visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]
plt.bar(range(len(missing_before)), missing_before.values, color='red', alpha=0.7)
plt.xticks(range(len(missing_before)), missing_before.index, rotation=45)
plt.title('Missing Values - BEFORE Cleaning')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
missing_after = df_clean.isnull().sum()
missing_after = missing_after[missing_after > 0]
if len(missing_after) > 0:
    plt.bar(range(len(missing_after)), missing_after.values, color='green', alpha=0.7)
    plt.xticks(range(len(missing_after)), missing_after.index, rotation=45)
else:
    plt.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
             fontsize=16, fontweight='bold', color='green')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
plt.title('Missing Values - AFTER Cleaning')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Save cleaned dataset info
print(f"\n💾 Cleaned Dataset Summary:")
print(f"   Shape: {df_clean.shape}")
print(f"   Memory usage: {df_clean.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"   Data types: {df_clean.dtypes.value_counts().to_dict()}")

print("\n" + "="*60)
print("✅ STEP 1 COMPLETED: DATA CLEANING AND MISSING VALUES")
print("✅ Dataset is now ready for further analysis!")
print("="*60)
