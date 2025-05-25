import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming netflix_df is already loaded from Step 1
# If running separately, uncomment the next line and run Step 1 first
# exec(open('step1_data_setup.py').read())

print("="*70)
print("STEP 2: DATA INTEGRITY AND CONSISTENCY CHECK")
print("="*70)

def check_data_integrity(df):
    """Comprehensive data integrity and consistency check"""
    
    integrity_report = {}
    
    print("🔍 CHECKING DATA INTEGRITY AND CONSISTENCY...")
    print("\n" + "="*50)
    print("1. MISSING VALUES ANALYSIS")
    print("="*50)
    
    # Missing values analysis
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_percentage
    }).sort_values('Missing_Count', ascending=False)
    
    print(missing_df[missing_df['Missing_Count'] > 0])
    integrity_report['missing_values'] = missing_df
    
    # Visualize missing data
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    missing_df[missing_df['Missing_Count'] > 0]['Missing_Count'].plot(kind='bar', color='coral')
    plt.title('Missing Values Count by Column')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    missing_df[missing_df['Missing_Count'] > 0]['Missing_Percentage'].plot(kind='bar', color='lightblue')
    plt.title('Missing Values Percentage by Column')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("2. DATA TYPES CONSISTENCY")
    print("="*50)
    
    # Data types check
    print("Current Data Types:")
    print(df.dtypes)
    
    # Check for inconsistent data types
    print("\n🔧 DATA TYPE CORRECTIONS NEEDED:")
    
    # Check if Release_Year should be integer
    if df['Release_Year'].dtype != 'int64':
        print("- Release_Year should be integer type")
    
    # Check if IMDb_Rating is in valid range
    rating_issues = df[(df['IMDb_Rating'] < 1) | (df['IMDb_Rating'] > 10)].dropna()
    if not rating_issues.empty:
        print(f"- Found {len(rating_issues)} IMDb ratings outside valid range (1-10)")
        print(rating_issues[['Title', 'IMDb_Rating']])
    
    print("\n" + "="*50)
    print("3. DUPLICATE RECORDS CHECK")
    print("="*50)
    
    # Check for duplicates
    duplicates = df.duplicated()
    duplicate_count = duplicates.sum()
    
    print(f"Total duplicate rows: {duplicate_count}")
    
    if duplicate_count > 0:
        print("Duplicate rows found:")
        print(df[duplicates])
    else:
        print("✓ No duplicate rows found")
    
    integrity_report['duplicates'] = duplicate_count
    
    print("\n" + "="*50)
    print("4. LOGICAL CONSISTENCY CHECKS")
    print("="*50)
    
    # Check logical consistency
    consistency_issues = []
    
    # Check if Seasons and Episodes_Total make sense
    unrealistic_episodes = df[df['Episodes_Total'] < df['Seasons']]
    if not unrealistic_episodes.empty:
        issue = f"Found {len(unrealistic_episodes)} shows with fewer episodes than seasons"
        consistency_issues.append(issue)
        print(f"⚠️  {issue}")
    
    # Check if Release_Year is reasonable for Netflix
    old_shows = df[df['Release_Year'] < 2010]
    if not old_shows.empty:
        issue = f"Found {len(old_shows)} shows released before 2010 (pre-Netflix originals era)"
        consistency_issues.append(issue)
        print(f"⚠️  {issue}")
    
    # Check if Runtime is reasonable
    unrealistic_runtime = df[(df['Runtime_Minutes'] < 5) | (df['Runtime_Minutes'] > 300)]
    if not unrealistic_runtime.empty:
        issue = f"Found {len(unrealistic_runtime)} shows with unrealistic runtime"
        consistency_issues.append(issue)
        print(f"⚠️  {issue}")
        print(unrealistic_runtime[['Title', 'Runtime_Minutes']].head())
    
    if not consistency_issues:
        print("✓ No major logical consistency issues found")
    
    integrity_report['consistency_issues'] = consistency_issues
    
    print("\n" + "="*50)
    print("5. VALUE RANGE VALIDATION")
    print("="*50)
    
    # Value range validation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        print(f"\n{col}:")
        print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f}")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        
        # Check for negative values where they shouldn't exist
        if col in ['IMDb_Votes', 'Seasons', 'Episodes_Total', 'Production_Budget_Million', 'Runtime_Minutes']:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"  ⚠️  Found {negative_count} negative values (should be positive)")
    
    print("\n" + "="*50)
    print("6. CATEGORICAL DATA VALIDATION")
    print("="*50)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        unique_count = df[col].nunique()
        print(f"\n{col}: {unique_count} unique values")
        print(f"Values: {sorted(df[col].unique())}")
        
        # Check for potential data entry issues
        if col == 'Genre':
            # Check for mixed case or unusual entries
            genres = df[col].unique()
            print(f"  Genres found: {list(genres)}")
    
    return integrity_report

# Run integrity check
print("Starting comprehensive data integrity check...\n")
integrity_results = check_data_integrity(netflix_df)

print("\n" + "="*70)
print("STEP 2 SUMMARY:")
print("="*70)
print("✓ Data integrity check completed")
print("✓ Missing values identified and quantified")
print("✓ Data types consistency verified")
print("✓ Duplicate records checked")
print("✓ Logical consistency validated")
print("✓ Value ranges examined")
print("✓ Categorical data validated")

print("\n🎯 KEY FINDINGS:")
missing_cols = integrity_results['missing_values'][integrity_results['missing_values']['Missing_Count'] > 0]
if not missing_cols.empty:
    print(f"- Missing data found in {len(missing_cols)} columns")
    for col in missing_cols.index:
        print(f"  • {col}: {missing_cols.loc[col, 'Missing_Count']} missing ({missing_cols.loc[col, 'Missing_Percentage']}%)")

if integrity_results['duplicates'] > 0:
    print(f"- {integrity_results['duplicates']} duplicate records found")
else:
    print("- No duplicate records found")

if integrity_results['consistency_issues']:
    print("- Consistency issues identified:")
    for issue in integrity_results['consistency_issues']:
        print(f"  • {issue}")

print("\n✓ Step 2 Complete: Data integrity analysis finished!")