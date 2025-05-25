import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

print("=" * 60)
print("STEP 2: FEATURE SELECTION AND ENGINEERING")
print("=" * 60)

# ============================================================================
# LOAD CLEANED DATA (From Step 1)
# ============================================================================

print("\n📊 Loading cleaned Netflix dataset...")

# Recreate the cleaned dataset from Step 1
np.random.seed(42)
n_samples = 500

# Generate the same dataset structure
titles = [f"Netflix Original {i}" for i in range(1, n_samples + 1)]
genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Horror', 'Romance', 'Sci-Fi', 
          'Documentary', 'Animation', 'Crime']
languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 
             'Japanese', 'Korean', 'Hindi', 'Mandarin']
content_types = ['Movie', 'TV Series', 'Limited Series', 'Documentary']

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

df = pd.DataFrame(data)
df['IMDb_Rating'] = df['IMDb_Rating'].clip(1, 10)
df['Cast_Rating'] = df['Cast_Rating'].clip(1, 10)
df['Release_Year'] = df['Release_Date'].dt.year

print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
print(f"📋 Original features: {list(df.columns)}")

# ============================================================================
# FEATURE ENGINEERING - TEMPORAL FEATURES
# ============================================================================

print("\n🕒 TEMPORAL FEATURE ENGINEERING")
print("-" * 35)

# Extract temporal features
df['Release_Month'] = df['Release_Date'].dt.month
df['Release_Quarter'] = df['Release_Date'].dt.quarter
df['Release_Day_of_Week'] = df['Release_Date'].dt.dayofweek
df['Release_Day_Name'] = df['Release_Date'].dt.day_name()
df['Age_Years'] = 2024 - df['Release_Year']
df['Is_Recent'] = (df['Age_Years'] <= 2).astype(int)
df['Decade'] = ((df['Release_Year'] // 10) * 10).astype(str) + 's'

# Season feature
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall'}
df['Release_Season'] = df['Release_Month'].map(season_map)

temporal_features = ['Release_Month', 'Release_Quarter', 'Release_Day_of_Week', 
                    'Age_Years', 'Is_Recent', 'Release_Season']

print(f"✅ Created {len(temporal_features)} temporal features:")
for feature in temporal_features:
    print(f"   - {feature}")

# ============================================================================
# FEATURE ENGINEERING - CATEGORICAL TRANSFORMATIONS
# ============================================================================

print("\n🏷️ CATEGORICAL FEATURE ENGINEERING")
print("-" * 38)

# Create categorical bins for continuous variables
df['Runtime_Category'] = pd.cut(df['Runtime_Minutes'], 
                               bins=[0, 90, 120, 180, float('inf')], 
                               labels=['Short', 'Medium', 'Long', 'Very_Long'])

df['Budget_Category'] = pd.qcut(df['Budget_Million_USD'], 
                               q=4, 
                               labels=['Low_Budget', 'Medium_Budget', 'High_Budget', 'Premium_Budget'])

df['Experience_Level'] = pd.cut(df['Director_Experience_Years'], 
                               bins=[0, 3, 8, 15, float('inf')], 
                               labels=['Newcomer', 'Experienced', 'Veteran', 'Master'])

df['Popularity_Tier'] = pd.qcut(df['IMDb_Votes'], 
                               q=3, 
                               labels=['Niche', 'Popular', 'Viral'])

# Language grouping (English vs Non-English)
df['Language_Group'] = df['Language'].apply(lambda x: 'English' if x == 'English' else 'Non_English')

# Genre grouping
high_budget_genres = ['Action', 'Sci-Fi', 'Thriller']
df['Genre_Budget_Expectation'] = df['Genre'].apply(
    lambda x: 'High_Budget_Genre' if x in high_budget_genres else 'Standard_Genre'
)

categorical_features = ['Runtime_Category', 'Budget_Category', 'Experience_Level', 
                       'Popularity_Tier', 'Language_Group', 'Genre_Budget_Expectation']

print(f"✅ Created {len(categorical_features)} categorical features:")
for feature in categorical_features:
    unique_vals = df[feature].nunique()
    print(f"   - {feature}: {unique_vals} categories")

# ============================================================================
# FEATURE ENGINEERING - NUMERICAL TRANSFORMATIONS
# ============================================================================

print("\n🔢 NUMERICAL FEATURE ENGINEERING")
print("-" * 35)

# Ratio and interaction features
df['Budget_per_Minute'] = df['Budget_Million_USD'] / df['Runtime_Minutes']
df['Views_per_Dollar'] = df['Netflix_Views_Million'] / df['Budget_Million_USD']
df['Rating_Popularity_Score'] = df['IMDb_Rating'] * np.log1p(df['IMDb_Votes'])
df['Experience_Budget_Ratio'] = df['Director_Experience_Years'] / (df['Budget_Million_USD'] + 1)
df['Cast_Runtime_Index'] = df['Cast_Rating'] * np.sqrt(df['Runtime_Minutes'])

# Polynomial features
df['Budget_Squared'] = df['Budget_Million_USD'] ** 2
df['Rating_Squared'] = df['IMDb_Rating'] ** 2

# Log transformations for skewed features
df['Budget_Log'] = np.log1p(df['Budget_Million_USD'])
df['Views_Log'] = np.log1p(df['Netflix_Views_Million'])
df['Votes_Log'] = np.log1p(df['IMDb_Votes'])

# Standardized features
scaler = StandardScaler()
numerical_cols = ['Runtime_Minutes', 'Director_Experience_Years', 'Cast_Rating']
df[['Runtime_Scaled', 'Experience_Scaled', 'Cast_Scaled']] = scaler.fit_transform(df[numerical_cols])

numerical_features = ['Budget_per_Minute', 'Views_per_Dollar', 'Rating_Popularity_Score',
                     'Experience_Budget_Ratio', 'Cast_Runtime_Index', 'Budget_Squared',
                     'Rating_Squared', 'Budget_Log', 'Views_Log', 'Votes_Log']

print(f"✅ Created {len(numerical_features)} numerical features:")
for feature in numerical_features:
    print(f"   - {feature}")

# ============================================================================
# FEATURE ENCODING
# ============================================================================

print("\n🔤 FEATURE ENCODING")
print("-" * 20)

# Label encoding for categorical variables
label_encoders = {}
categorical_cols_to_encode = ['Genre', 'Language', 'Content_Type', 'Release_Season', 
                             'Runtime_Category', 'Experience_Level', 'Language_Group']

for col in categorical_cols_to_encode:
    le = LabelEncoder()
    df[f'{col}_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-hot encoding for selected features
one_hot_features = ['Content_Type', 'Genre']
df_encoded = pd.get_dummies(df, columns=one_hot_features, prefix=one_hot_features)

print(f"✅ Label encoded: {len(categorical_cols_to_encode)} features")
print(f"✅ One-hot encoded: {len(one_hot_features)} features")
print(f"📊 Total features after encoding: {df_encoded.shape[1]}")

# ============================================================================
# FEATURE SELECTION - CORRELATION ANALYSIS
# ============================================================================

print("\n📈 FEATURE SELECTION - CORRELATION ANALYSIS")
print("-" * 45)

# Select numerical features for correlation analysis
numerical_features_for_corr = ['Runtime_Minutes', 'Budget_Million_USD', 'IMDb_Votes',
                              'Netflix_Views_Million', 'Director_Experience_Years', 'Cast_Rating',
                              'Age_Years', 'Budget_per_Minute', 'Views_per_Dollar', 
                              'Rating_Popularity_Score', 'IMDb_Rating']

correlation_matrix = df[numerical_features_for_corr].corr()

# Find features highly correlated with target (IMDb_Rating)
target_correlations = correlation_matrix['IMDb_Rating'].abs().sort_values(ascending=False)
print("Top 10 features correlated with IMDb Rating:")
print("+" + "-" * 50 + "+")
for i, (feature, corr) in enumerate(target_correlations.head(10).items()):
    if feature != 'IMDb_Rating':
        print(f"| {i:2d}. {feature:<35} | {corr:>6.3f} |")
print("+" + "-" * 50 + "+")

# Visualize correlation matrix
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Identify multicollinear features
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = abs(correlation_matrix.iloc[i, j])
        if corr_val > 0.8 and correlation_matrix.columns[i] != 'IMDb_Rating' and correlation_matrix.columns[j] != 'IMDb_Rating':
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))

print(f"\n🚨 High correlation pairs (|r| > 0.8): {len(high_corr_pairs)}")
for feat1, feat2, corr in high_corr_pairs:
    print(f"   - {feat1} ↔ {feat2}: {corr:.3f}")

# ============================================================================
# FEATURE SELECTION - STATISTICAL METHODS
# ============================================================================

print("\n📊 FEATURE SELECTION - STATISTICAL METHODS")
print("-" * 42)

# Prepare features for statistical selection
feature_columns = [col for col in df.columns if col not in ['Title', 'Release_Date', 'IMDb_Rating']]
numerical_feature_columns = df[feature_columns].select_dtypes(include=[np.number]).columns

X = df[numerical_feature_columns].fillna(0)
y = df['IMDb_Rating']

# F-statistic based selection
f_selector = SelectKBest(score_func=f_regression, k=10)
X_f_selected = f_selector.fit_transform(X, y)
f_selected_features = X.columns[f_selector.get_support()]
f_scores = f_selector.scores_[f_selector.get_support()]

print("Top 10 features by F-statistic:")
print("+" + "-" * 45 + "+")
for i, (feature, score) in enumerate(zip(f_selected_features, f_scores)):
    print(f"| {i+1:2d}. {feature:<30} | {score:>8.2f} |")
print("+" + "-" * 45 + "+")

# Mutual Information based selection
mi_selector = SelectKBest(score_func=mutual_info_regression, k=10)
X_mi_selected = mi_selector.fit_transform(X, y)
mi_selected_features = X.columns[mi_selector.get_support()]
mi_scores = mi_selector.scores_[mi_selector.get_support()]

print("\nTop 10 features by Mutual Information:")
print("+" + "-" * 45 + "+")
for i, (feature, score) in enumerate(zip(mi_selected_features, mi_scores)):
    print(f"| {i+1:2d}. {feature:<30} | {score:>8.3f} |")
print("+" + "-" * 45 + "+")

# ============================================================================
# DIMENSIONALITY REDUCTION - PCA
# ============================================================================

print("\n🔍 DIMENSIONALITY REDUCTION - PCA")
print("-" * 32)

# Apply PCA to numerical features
pca = PCA()
X_scaled = StandardScaler().fit_transform(X)
pca_result = pca.fit_transform(X_scaled)

# Calculate cumulative explained variance
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
n_components_90 = np.argmax(cumsum_var >= 0.90) + 1

print(f"📊 PCA Results:")
print(f"   - Components for 90% variance: {n_components_90}")
print(f"   - Components for 95% variance: {n_components_95}")
print(f"   - Total original features: {X.shape[1]}")

# Visualize PCA results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(range(1, len(cumsum_var) + 1), cumsum_var * 100, 'ro-')
plt.axhline(y=90, color='g', linestyle='--', label='90%')
plt.axhline(y=95, color='b', linestyle='--', label='95%')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Feature importance in first two components
feature_importance = np.abs(pca.components_[:2]).mean(axis=0)
top_features_idx = np.argsort(feature_importance)[-10:]
plt.barh(range(10), feature_importance[top_features_idx])
plt.yticks(range(10), X.columns[top_features_idx])
plt.xlabel('Average Absolute Loading')
plt.title('Top 10 Features in PC1-PC2')

plt.tight_layout()
plt.show()

# ============================================================================
# FINAL FEATURE SELECTION SUMMARY
# ============================================================================

print("\n🎯 FINAL FEATURE SELECTION SUMMARY")
print("-" * 36)

# Combine selection methods to get final feature set
correlation_features = set(target_correlations.head(8).index) - {'IMDb_Rating'}
f_stat_features = set(f_selected_features)
mi_features = set(mi_selected_features)

# Features that appear in multiple selection methods
consensus_features = correlation_features.intersection(f_stat_features).intersection(mi_features)
recommended_features = correlation_features.union(f_stat_features).union(mi_features)

print(f"📊 Feature Selection Results:")
print(f"   - High correlation features: {len(correlation_features)}")
print(f"   - F-statistic selected: {len(f_stat_features)}")
print(f"   - Mutual info selected: {len(mi_features)}")
print(f"   - Consensus features: {len(consensus_features)}")
print(f"   - Recommended features: {len(recommended_features)}")

print(f"\n🏆 CONSENSUS FEATURES (appear in all methods):")
for i, feature in enumerate(sorted(consensus_features), 1):
    print(f"   {i:2d}. {feature}")

print(f"\n✨ RECOMMENDED FINAL FEATURE SET:")
for i, feature in enumerate(sorted(recommended_features), 1):
    print(f"   {i:2d}. {feature}")

# Create final dataset with selected features
final_features = list(recommended_features) + ['IMDb_Rating', 'Title', 'Genre', 'Content_Type']
df_final = df[final_features].copy()

print(f"\n📋 Final Dataset Shape: {df_final.shape}")
print(f"📈 Feature Engineering Summary:")
print(f"   - Original features: {len(data.keys())}")
print(f"   - Created features: {df.shape[1] - len(data.keys())}")
print(f"   - Final selected: {len(final_features) - 3}")  # Excluding target and identifiers

# Feature importance visualization
plt.figure(figsize=(12, 8))

# Plot feature importance scores
feature_scores = {}
for feature in recommended_features:
    if feature in target_correlations.index:
        feature_scores[feature] = target_correlations[feature]

sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
features, scores = zip(*sorted_features)

plt.barh(range(len(features)), scores, color='skyblue', edgecolor='navy', alpha=0.7)
plt.yticks(range(len(features)), features)
plt.xlabel('Correlation with IMDb Rating')
plt.title('Feature Importance Ranking')
plt.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, score in enumerate(scores):
    plt.text(score + 0.01 if score >= 0 else score - 0.01, i, f'{score:.3f}', 
             va='center', ha='left' if score >= 0 else 'right')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("✅ STEP 2 COMPLETED: FEATURE SELECTION AND ENGINEERING")
print("✅ Engineered features ready for analysis!")
print("="*60)

# Store the engineered dataset for next steps
engineered_netflix_data = df_final.copy()
feature_engineering_summary = {
    'original_features': len(data.keys()),
    'total_created': df.shape[1] - len(data.keys()),
    'final_selected': len(final_features) - 3,
    'consensus_features': list(consensus_features),
    'recommended_features': list(recommended_features)
}