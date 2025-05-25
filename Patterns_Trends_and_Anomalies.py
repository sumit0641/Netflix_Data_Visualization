import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetflixPatternAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for pattern analysis"""
        # Convert date columns if they exist
        if 'Premiere' in self.df.columns:
            self.df['Premiere'] = pd.to_datetime(self.df['Premiere'], errors='coerce')
            self.df['Year'] = self.df['Premiere'].dt.year
            self.df['Month'] = self.df['Premiere'].dt.month
        
        # Ensure IMDB rating is numeric
        if 'IMDB Score' in self.df.columns:
            self.df['IMDB Score'] = pd.to_numeric(self.df['IMDB Score'], errors='coerce')
    
    def temporal_trends(self):
        """Analyze temporal patterns in Netflix releases and ratings"""
        print("=== TEMPORAL TRENDS ANALYSIS ===\n")
        
        # 1. Release trends over time
        if 'Year' in self.df.columns:
            yearly_releases = self.df.groupby('Year').size()
            yearly_avg_rating = self.df.groupby('Year')['IMDB Score'].mean()
            
            print("📈 Release Volume Trends:")
            print(f"   • Peak release year: {yearly_releases.idxmax()} ({yearly_releases.max()} releases)")
            print(f"   • Lowest release year: {yearly_releases.idxmin()} ({yearly_releases.min()} releases)")
            print(f"   • Average releases per year: {yearly_releases.mean():.1f}")
            
            # Trend analysis
            years = yearly_releases.index
            releases = yearly_releases.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, releases)
            
            trend_direction = "increasing" if slope > 0 else "decreasing"
            print(f"   • Overall trend: {trend_direction} (slope: {slope:.2f}, R²: {r_value**2:.3f})")
            
            print(f"\n⭐ Rating Trends Over Time:")
            print(f"   • Highest rated year: {yearly_avg_rating.idxmax()} (avg: {yearly_avg_rating.max():.2f})")
            print(f"   • Lowest rated year: {yearly_avg_rating.idxmin()} (avg: {yearly_avg_rating.min():.2f})")
            
        # 2. Seasonal patterns
        if 'Month' in self.df.columns:
            monthly_releases = self.df.groupby('Month').size()
            monthly_ratings = self.df.groupby('Month')['IMDB Score'].mean()
            
            print(f"\n📅 Seasonal Release Patterns:")
            print(f"   • Peak release month: {monthly_releases.idxmax()} ({monthly_releases.max()} releases)")
            print(f"   • Lowest release month: {monthly_releases.idxmin()} ({monthly_releases.min()} releases)")
    
    def genre_patterns(self):
        """Analyze patterns by genre"""
        print("\n=== GENRE PATTERNS ANALYSIS ===\n")
        
        if 'Genre' in self.df.columns:
            # Genre performance
            genre_stats = self.df.groupby('Genre').agg({
                'IMDB Score': ['mean', 'std', 'count'],
                'Runtime': 'mean' if 'Runtime' in self.df.columns else lambda x: None
            }).round(2)
            
            genre_ratings = self.df.groupby('Genre')['IMDB Score'].mean().sort_values(ascending=False)
            
            print("🎭 Genre Performance Rankings:")
            for i, (genre, rating) in enumerate(genre_ratings.head(5).items(), 1):
                count = self.df[self.df['Genre'] == genre].shape[0]
                print(f"   {i}. {genre}: {rating:.2f} avg rating ({count} titles)")
            
            print(f"\n📊 Genre Distribution:")
            genre_counts = self.df['Genre'].value_counts()
            for i, (genre, count) in enumerate(genre_counts.head(5).items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"   {i}. {genre}: {count} titles ({pct:.1f}%)")
    
    def rating_distribution_patterns(self):
        """Analyze rating distribution patterns"""
        print("\n=== RATING DISTRIBUTION PATTERNS ===\n")
        
        ratings = self.df['IMDB Score'].dropna()
        
        # Basic statistics
        print("📈 Rating Statistics:")
        print(f"   • Mean: {ratings.mean():.2f}")
        print(f"   • Median: {ratings.median():.2f}")
        print(f"   • Standard Deviation: {ratings.std():.2f}")
        print(f"   • Range: {ratings.min():.1f} - {ratings.max():.1f}")
        
        # Quartiles and percentiles
        q1, q3 = ratings.quantile([0.25, 0.75])
        print(f"   • Q1 (25th percentile): {q1:.2f}")
        print(f"   • Q3 (75th percentile): {q3:.2f}")
        print(f"   • IQR: {q3 - q1:.2f}")
        
        # Distribution characteristics
        skewness = stats.skew(ratings)
        kurtosis = stats.kurtosis(ratings)
        
        print(f"\n📊 Distribution Shape:")
        print(f"   • Skewness: {skewness:.3f} ({'right-skewed' if skewness > 0 else 'left-skewed' if skewness < 0 else 'symmetric'})")
        print(f"   • Kurtosis: {kurtosis:.3f} ({'heavy-tailed' if kurtosis > 0 else 'light-tailed' if kurtosis < 0 else 'normal-tailed'})")
        
        # Rating categories
        excellent = (ratings >= 8.0).sum()
        good = ((ratings >= 7.0) & (ratings < 8.0)).sum()
        average = ((ratings >= 6.0) & (ratings < 7.0)).sum()
        poor = (ratings < 6.0).sum()
        
        total = len(ratings)
        print(f"\n⭐ Rating Categories:")
        print(f"   • Excellent (8.0+): {excellent} titles ({excellent/total*100:.1f}%)")
        print(f"   • Good (7.0-7.9): {good} titles ({good/total*100:.1f}%)")
        print(f"   • Average (6.0-6.9): {average} titles ({average/total*100:.1f}%)")
        print(f"   • Poor (<6.0): {poor} titles ({poor/total*100:.1f}%)")
    
    def correlation_patterns(self):
        """Identify correlation patterns between variables"""
        print("\n=== CORRELATION PATTERNS ===\n")
        
        # Select numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        print("🔗 Strong Correlations (|r| > 0.5):")
        
        # Find strong correlations
        strong_corrs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if strong_corrs:
            for corr in sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True):
                direction = "positive" if corr['correlation'] > 0 else "negative"
                print(f"   • {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f} ({direction})")
        else:
            print("   • No strong correlations found (|r| > 0.5)")
    
    def identify_anomalies(self):
        """Identify potential anomalies in the data"""
        print("\n=== ANOMALY DETECTION ===\n")
        
        ratings = self.df['IMDB Score'].dropna()
        
        # Statistical outliers using IQR method
        Q1 = ratings.quantile(0.25)
        Q3 = ratings.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ratings[(ratings < lower_bound) | (ratings > upper_bound)]
        
        print(f"📊 Statistical Outliers (IQR method):")
        print(f"   • Lower bound: {lower_bound:.2f}")
        print(f"   • Upper bound: {upper_bound:.2f}")
        print(f"   • Number of outliers: {len(outliers)}")
        print(f"   • Percentage of data: {len(outliers)/len(ratings)*100:.1f}%")
        
        if len(outliers) > 0:
            print(f"   • Outlier range: {outliers.min():.2f} - {outliers.max():.2f}")
        
        # Z-score method
        z_scores = np.abs(stats.zscore(ratings))
        z_outliers = ratings[z_scores > 3]
        
        print(f"\n📈 Z-score Outliers (|z| > 3):")
        print(f"   • Number of outliers: {len(z_outliers)}")
        print(f"   • Percentage of data: {len(z_outliers)/len(ratings)*100:.1f}%")
    
    def run_full_analysis(self):
        """Run complete pattern analysis"""
        print("🎬 NETFLIX ORIGINALS - PATTERN & TREND ANALYSIS")
        print("=" * 60)
        
        self.temporal_trends()
        self.genre_patterns()
        self.rating_distribution_patterns()
        self.correlation_patterns()
        self.identify_anomalies()
        
        print("\n" + "=" * 60)
        print("✅ Pattern analysis completed!")

# Example usage with sample data
if __name__ == "__main__":
    # Create sample Netflix data for demonstration
    np.random.seed(42)
    
    sample_data = {
        'Title': [f'Netflix Original {i}' for i in range(1, 101)],
        'Genre': np.random.choice(['Drama', 'Comedy', 'Action', 'Documentary', 'Horror', 'Romance'], 100),
        'IMDB Score': np.random.normal(6.8, 1.2, 100).clip(3.0, 9.5),
        'Runtime': np.random.normal(105, 25, 100).clip(80, 180),
        'Premiere': pd.date_range('2015-01-01', '2024-12-31', periods=100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analyzer
    analyzer = NetflixPatternAnalyzer(df)
    
    # Run analysis
    analyzer.run_full_analysis()