import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class NetflixOutlierHandler:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.outlier_info = {}
        
    def detect_outliers_iqr(self, column, multiplier=1.5):
        """Detect outliers using IQR method"""
        data = self.df[column].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_indices = data[(data < lower_bound) | (data > upper_bound)].index
        
        return {
            'method': 'IQR',
            'column': column,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers': outliers,
            'outlier_indices': outlier_indices,
            'n_outliers': len(outliers),
            'percentage': len(outliers) / len(data) * 100
        }
    
    def detect_outliers_zscore(self, column, threshold=3):
        """Detect outliers using Z-score method"""
        data = self.df[column].dropna()
        z_scores = np.abs(stats.zscore(data))
        
        outliers = data[z_scores > threshold]
        outlier_indices = data[z_scores > threshold].index
        
        return {
            'method': 'Z-Score',
            'column': column,
            'threshold': threshold,
            'outliers': outliers,
            'outlier_indices': outlier_indices,
            'n_outliers': len(outliers),
            'percentage': len(outliers) / len(data) * 100
        }
    
    def detect_outliers_modified_zscore(self, column, threshold=3.5):
        """Detect outliers using Modified Z-score (MAD-based)"""
        data = self.df[column].dropna()
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        modified_z_scores = np.abs(0.6745 * (data - median) / mad)
        outliers = data[modified_z_scores > threshold]
        outlier_indices = data[modified_z_scores > threshold].index
        
        return {
            'method': 'Modified Z-Score',
            'column': column,
            'threshold': threshold,
            'outliers': outliers,
            'outlier_indices': outlier_indices,
            'n_outliers': len(outliers),
            'percentage': len(outliers) / len(data) * 100
        }
    
    def analyze_outliers(self, columns, methods=['iqr', 'zscore']):
        """Comprehensive outlier analysis"""
        print("🔍 OUTLIER DETECTION ANALYSIS")
        print("=" * 50)
        
        for column in columns:
            if column not in self.df.columns:
                print(f"⚠️ Column '{column}' not found in dataset")
                continue
                
            print(f"\n📊 Analyzing column: {column}")
            print("-" * 30)
            
            # Basic statistics
            data = self.df[column].dropna()
            print(f"Data points: {len(data)}")
            print(f"Mean: {data.mean():.3f}")
            print(f"Median: {data.median():.3f}")
            print(f"Std: {data.std():.3f}")
            print(f"Min: {data.min():.3f}, Max: {data.max():.3f}")
            
            # Apply different methods
            outlier_results = {}
            
            if 'iqr' in methods:
                iqr_result = self.detect_outliers_iqr(column)
                outlier_results['IQR'] = iqr_result
                print(f"\n🎯 IQR Method:")
                print(f"   Bounds: [{iqr_result['lower_bound']:.3f}, {iqr_result['upper_bound']:.3f}]")
                print(f"   Outliers: {iqr_result['n_outliers']} ({iqr_result['percentage']:.1f}%)")
            
            if 'zscore' in methods:
                zscore_result = self.detect_outliers_zscore(column)
                outlier_results['Z-Score'] = zscore_result
                print(f"\n📈 Z-Score Method:")
                print(f"   Threshold: {zscore_result['threshold']}")
                print(f"   Outliers: {zscore_result['n_outliers']} ({zscore_result['percentage']:.1f}%)")
            
            if 'modified_zscore' in methods:
                mod_zscore_result = self.detect_outliers_modified_zscore(column)
                outlier_results['Modified Z-Score'] = mod_zscore_result
                print(f"\n🎲 Modified Z-Score Method:")
                print(f"   Threshold: {mod_zscore_result['threshold']}")
                print(f"   Outliers: {mod_zscore_result['n_outliers']} ({mod_zscore_result['percentage']:.1f}%)")
            
            self.outlier_info[column] = outlier_results
    
    def handle_outliers(self, column, method='cap', detection_method='iqr'):
        """Handle outliers using various strategies"""
        if column not in self.outlier_info:
            self.analyze_outliers([column], [detection_method])
        
        outlier_data = self.outlier_info[column][detection_method.upper()]
        
        print(f"\n🛠️ HANDLING OUTLIERS - {column}")
        print(f"Method: {method.upper()}")
        print("-" * 30)
        
        original_count = len(self.df[column].dropna())
        
        if method == 'remove':
            # Remove outliers
            self.df = self.df.drop(outlier_data['outlier_indices'])
            new_count = len(self.df[column].dropna())
            print(f"✂️ Removed {original_count - new_count} outliers")
            
        elif method == 'cap':
            # Cap outliers to bounds
            if 'lower_bound' in outlier_data:
                self.df.loc[self.df[column] < outlier_data['lower_bound'], column] = outlier_data['lower_bound']
                self.df.loc[self.df[column] > outlier_data['upper_bound'], column] = outlier_data['upper_bound']
                print(f"🧢 Capped outliers to bounds: [{outlier_data['lower_bound']:.3f}, {outlier_data['upper_bound']:.3f}]")
        
        elif method == 'transform':
            # Log transformation to reduce impact
            if (self.df[column] > 0).all():
                self.df[column + '_log'] = np.log(self.df[column])
                print(f"📊 Applied log transformation (new column: {column}_log)")
            else:
                print("⚠️ Cannot apply log transformation: non-positive values present")
        
        elif method == 'winsorize':
            # Winsorize at 5th and 95th percentiles
            p5 = self.df[column].quantile(0.05)
            p95 = self.df[column].quantile(0.95)
            self.df[column] = self.df[column].clip(lower=p5, upper=p95)
            print(f"🎯 Winsorized to 5th-95th percentiles: [{p5:.3f}, {p95:.3f}]")
    
    def apply_transformations(self, columns, transformations=['standard', 'minmax', 'robust']):
        """Apply various data transformations"""
        print("\n🔄 DATA TRANSFORMATIONS")
        print("=" * 50)
        
        transformed_data = {}
        
        for column in columns:
            if column not in self.df.columns:
                continue
                
            data = self.df[column].dropna().values.reshape(-1, 1)
            print(f"\n📊 Transforming: {column}")
            print("-" * 20)
            
            # Original statistics
            print(f"Original - Mean: {data.mean():.3f}, Std: {data.std():.3f}")
            print(f"Original - Min: {data.min():.3f}, Max: {data.max():.3f}")
            
            if 'standard' in transformations:
                # Standard Scaling (Z-score normalization)
                scaler = StandardScaler()
                standardized = scaler.fit_transform(data)
                transformed_data[f'{column}_standard'] = standardized.flatten()
                print(f"Standard - Mean: {standardized.mean():.3f}, Std: {standardized.std():.3f}")
            
            if 'minmax' in transformations:
                # Min-Max Scaling
                scaler = MinMaxScaler()
                minmax_scaled = scaler.fit_transform(data)
                transformed_data[f'{column}_minmax'] = minmax_scaled.flatten()
                print(f"MinMax - Min: {minmax_scaled.min():.3f}, Max: {minmax_scaled.max():.3f}")
            
            if 'robust' in transformations:
                # Robust Scaling (median and IQR)
                scaler = RobustScaler()
                robust_scaled = scaler.fit_transform(data)
                transformed_data[f'{column}_robust'] = robust_scaled.flatten()
                print(f"Robust - Median: {np.median(robust_scaled):.3f}, IQR: {np.percentile(robust_scaled, 75) - np.percentile(robust_scaled, 25):.3f}")
            
            if 'power' in transformations:
                # Power Transformation (Yeo-Johnson)
                try:
                    transformer = PowerTransformer(method='yeo-johnson')
                    power_transformed = transformer.fit_transform(data)
                    transformed_data[f'{column}_power'] = power_transformed.flatten()
                    print(f"Power - Skewness: {stats.skew(power_transformed):.3f}")
                except:
                    print("⚠️ Power transformation failed")
            
            if 'quantile' in transformations:
                # Quantile Transformation
                transformer = QuantileTransformer(output_distribution='uniform')
                quantile_transformed = transformer.fit_transform(data)
                transformed_data[f'{column}_quantile'] = quantile_transformed.flatten()
                print(f"Quantile - Range: [{quantile_transformed.min():.3f}, {quantile_transformed.max():.3f}]")
        
        # Add transformed columns to dataframe
        for col_name, values in transformed_data.items():
            # Match the length with original dataframe
            if len(values) == len(self.df):
                self.df[col_name] = values
            else:
                # Handle missing values
                temp_series = pd.Series(index=self.df.index, dtype=float)
                temp_series.iloc[:len(values)] = values
                self.df[col_name] = temp_series
        
        return transformed_data
    
    def distribution_analysis(self, columns):
        """Analyze distributions before and after transformations"""
        print("\n📈 DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        for column in columns:
            if column not in self.df.columns:
                continue
                
            data = self.df[column].dropna()
            
            print(f"\n📊 {column} Distribution:")
            print("-" * 25)
            print(f"Skewness: {stats.skew(data):.3f}")
            print(f"Kurtosis: {stats.kurtosis(data):.3f}")
            
            # Normality tests
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
                print(f"Shapiro-Wilk test p-value: {shapiro_p:.6f}")
                print(f"Normal distribution: {'No' if shapiro_p < 0.05 else 'Possibly'}")
            except:
                print("Shapiro-Wilk test: Could not be performed")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n📋 OUTLIER HANDLING SUMMARY REPORT")
        print("=" * 60)
        
        print(f"Original dataset shape: {self.original_df.shape}")
        print(f"Current dataset shape: {self.df.shape}")
        
        if self.outlier_info:
            print(f"\n🎯 Outlier Analysis Summary:")
            for column, methods in self.outlier_info.items():
                print(f"\n• {column}:")
                for method, info in methods.items():
                    print(f"  - {method}: {info['n_outliers']} outliers ({info['percentage']:.1f}%)")
        
        # List all transformations applied
        original_cols = set(self.original_df.columns)
        new_cols = set(self.df.columns) - original_cols
        
        if new_cols:
            print(f"\n🔄 Transformations Applied:")
            for col in sorted(new_cols):
                print(f"  • {col}")
        
        print(f"\n✅ Analysis completed successfully!")

# Example usage
if __name__ == "__main__":
    # Create sample Netflix data with outliers
    np.random.seed(42)
    
    # Generate data with some outliers
    normal_ratings = np.random.normal(6.8, 1.0, 90)
    outlier_ratings = [2.1, 2.5, 9.8, 9.9]  # Extreme outliers
    all_ratings = np.concatenate([normal_ratings, outlier_ratings])
    
    normal_runtime = np.random.normal(105, 15, 90)
    outlier_runtime = [45, 250, 300, 320]  # Runtime outliers
    all_runtime = np.concatenate([normal_runtime, outlier_runtime])
    
    sample_data = {
        'Title': [f'Netflix Original {i}' for i in range(1, 95)],
        'Genre': np.random.choice(['Drama', 'Comedy', 'Action', 'Documentary'], 94),
        'IMDB Score': all_ratings,
        'Runtime': all_runtime,
        'Year': np.random.choice(range(2015, 2025), 94)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize outlier handler
    handler = NetflixOutlierHandler(df)
    
    # Analyze outliers
    handler.analyze_outliers(['IMDB Score', 'Runtime'], methods=['iqr', 'zscore', 'modified_zscore'])
    
    # Handle outliers
    handler.handle_outliers('IMDB Score', method='cap', detection_method='iqr')
    handler.handle_outliers('Runtime', method='winsorize')
    
    # Apply transformations
    handler.apply_transformations(['IMDB Score', 'Runtime'], 
                                transformations=['standard', 'minmax', 'robust', 'power'])
    
    # Distribution analysis
    handler.distribution_analysis(['IMDB Score', 'Runtime'])
    
    # Generate summary report
    handler.generate_summary_report()