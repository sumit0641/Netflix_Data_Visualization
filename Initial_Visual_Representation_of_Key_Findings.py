import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('default')
sns.set_palette("Set2")

class NetflixVisualization:
    def __init__(self):
        """Initialize Netflix Visualization class"""
        self.df = self.create_sample_data()
        self.setup_plot_style()
    
    def create_sample_data(self):
        """Create realistic Netflix Originals dataset"""
        np.random.seed(42)
        
        # Define categories
        genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Documentary', 
                 'Horror', 'Romance', 'Sci-Fi', 'Crime', 'Animation']
        languages = ['English', 'Spanish', 'French', 'German', 'Korean', 
                    'Japanese', 'Hindi', 'Portuguese', 'Italian']
        
        n_samples = 300
        
        # Create realistic data with correlations
        base_ratings = np.random.normal(6.5, 1.3, n_samples)
        
        data = {
            'Title': [f'Netflix Original {i+1}' for i in range(n_samples)],
            'Genre': np.random.choice(genres, n_samples, 
                                    p=[0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.08, 0.06, 0.09, 0.07]),
            'Language': np.random.choice(languages, n_samples,
                                       p=[0.35, 0.15, 0.08, 0.06, 0.08, 0.06, 0.07, 0.05, 0.10]),
            'Runtime': np.clip(np.random.normal(105, 25, n_samples).astype(int), 60, 180),
            'IMDb_Rating': np.clip(base_ratings, 3.5, 9.5),
            'Release_Year': np.random.choice(range(2016, 2024), n_samples,
                                           p=[0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.15, 0.07]),
            'Number_of_Votes': np.random.lognormal(8, 1.5, n_samples).astype(int),
            'Budget_Category': np.random.choice(['Low', 'Medium', 'High'], n_samples,
                                              p=[0.4, 0.4, 0.2])
        }
        
        # Add some realistic correlations
        df = pd.DataFrame(data)
        
        # Higher budget tends to have slightly higher ratings
        budget_boost = {'Low': 0, 'Medium': 0.2, 'High': 0.4}
        df['IMDb_Rating'] += df['Budget_Category'].map(budget_boost)
        
        # Documentaries tend to have higher ratings
        df.loc[df['Genre'] == 'Documentary', 'IMDb_Rating'] += 0.3
        
        # Horror movies tend to have lower ratings
        df.loc[df['Genre'] == 'Horror', 'IMDb_Rating'] -= 0.2
        
        # Clip ratings to realistic range
        df['IMDb_Rating'] = np.clip(df['IMDb_Rating'], 3.5, 9.2)
        
        return df
    
    def setup_plot_style(self):
        """Setup consistent plot styling"""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard of key findings"""
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Netflix Originals IMDb Ratings Analysis - Key Findings Overview', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Rating Distribution
        plt.subplot(3, 3, 1)
        self.df['IMDb_Rating'].hist(bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.df['IMDb_Rating'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["IMDb_Rating"].mean():.2f}')
        plt.xlabel('IMDb Rating')
        plt.ylabel('Frequency')
        plt.title('Distribution of IMDb Ratings')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Ratings by Genre
        plt.subplot(3, 3, 2)
        genre_ratings = self.df.groupby('Genre')['IMDb_Rating'].mean().sort_values(ascending=True)
        genre_ratings.plot(kind='barh', color='lightcoral')
        plt.xlabel('Average IMDb Rating')
        plt.title('Average Rating by Genre')
        plt.grid(True, alpha=0.3)
        
        # 3. Ratings Over Time
        plt.subplot(3, 3, 3)
        yearly_ratings = self.df.groupby('Release_Year')['IMDb_Rating'].mean()
        plt.plot(yearly_ratings.index, yearly_ratings.values, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Release Year')
        plt.ylabel('Average IMDb Rating')
        plt.title('Rating Trends Over Time')
        plt.grid(True, alpha=0.3)
        
        # 4. Language Distribution
        plt.subplot(3, 3, 4)
        language_counts = self.df['Language'].value_counts().head(8)
        plt.pie(language_counts.values, labels=language_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Content Distribution by Language')
        
        # 5. Runtime vs Rating Scatter
        plt.subplot(3, 3, 5)
        plt.scatter(self.df['Runtime'], self.df['IMDb_Rating'], alpha=0.6, s=50)
        # Add trend line
        z = np.polyfit(self.df['Runtime'], self.df['IMDb_Rating'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['Runtime'], p(self.df['Runtime']), "r--", alpha=0.8)
        plt.xlabel('Runtime (minutes)')
        plt.ylabel('IMDb Rating')
        plt.title('Runtime vs IMDb Rating')
        plt.grid(True, alpha=0.3)
        
        # 6. Budget Category Impact
        plt.subplot(3, 3, 6)
        budget_ratings = self.df.groupby('Budget_Category')['IMDb_Rating'].mean()
        budget_order = ['Low', 'Medium', 'High']
        budget_ratings = budget_ratings.reindex(budget_order)
        bars = plt.bar(budget_ratings.index, budget_ratings.values, 
                      color=['lightblue', 'orange', 'lightgreen'])
        plt.xlabel('Budget Category')
        plt.ylabel('Average IMDb Rating')
        plt.title('Rating by Budget Category')
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
        
        # 7. Genre Popularity (Count)
        plt.subplot(3, 3, 7)
        genre_counts = self.df['Genre'].value_counts()
        genre_counts.plot(kind='bar', color='mediumpurple', alpha=0.8)
        plt.xlabel('Genre')
        plt.ylabel('Number of Titles')
        plt.title('Number of Titles by Genre')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 8. Rating Quality Categories
        plt.subplot(3, 3, 8)
        # Categorize ratings
        def categorize_rating(rating):
            if rating >= 8.0:
                return 'Excellent (8.0+)'
            elif rating >= 7.0:
                return 'Good (7.0-7.9)'
            elif rating >= 6.0:
                return 'Average (6.0-6.9)'
            else:
                return 'Below Average (<6.0)'
        
        self.df['Rating_Category'] = self.df['IMDb_Rating'].apply(categorize_rating)
        rating_cat_counts = self.df['Rating_Category'].value_counts()
        
        colors = ['gold', 'lightgreen', 'orange', 'lightcoral']
        plt.pie(rating_cat_counts.values, labels=rating_cat_counts.index, 
               autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Distribution of Rating Quality')
        
        # 9. Top Languages by Average Rating
        plt.subplot(3, 3, 9)
        lang_ratings = self.df.groupby('Language')['IMDb_Rating'].agg(['mean', 'count'])
        # Filter languages with at least 10 titles
        lang_ratings = lang_ratings[lang_ratings['count'] >= 10]
        lang_ratings = lang_ratings.sort_values('mean', ascending=True)
        
        plt.barh(range(len(lang_ratings)), lang_ratings['mean'], color='teal', alpha=0.7)
        plt.yticks(range(len(lang_ratings)), lang_ratings.index)
        plt.xlabel('Average IMDb Rating')
        plt.title('Average Rating by Language\n(Min 10 titles)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_detailed_findings_report(self):
        """Generate detailed statistical findings"""
        print("="*60)
        print("NETFLIX ORIGINALS - KEY FINDINGS SUMMARY")
        print("="*60)
        
        # Basic Statistics
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"Total Netflix Originals Analyzed: {len(self.df)}")
        print(f"Average IMDb Rating: {self.df['IMDb_Rating'].mean():.2f}")
        print(f"Rating Standard Deviation: {self.df['IMDb_Rating'].std():.2f}")
        print(f"Highest Rated: {self.df['IMDb_Rating'].max():.2f}")
        print(f"Lowest Rated: {self.df['IMDb_Rating'].min():.2f}")
        
        # Genre Analysis
        print(f"\n🎭 GENRE INSIGHTS:")
        genre_stats = self.df.groupby('Genre')['IMDb_Rating'].agg(['mean', 'count', 'std'])
        best_genre = genre_stats['mean'].idxmax()
        worst_genre = genre_stats['mean'].idxmin()
        print(f"Best Performing Genre: {best_genre} (Avg: {genre_stats.loc[best_genre, 'mean']:.2f})")
        print(f"Worst Performing Genre: {worst_genre} (Avg: {genre_stats.loc[worst_genre, 'mean']:.2f})")
        print(f"Most Produced Genre: {genre_stats['count'].idxmax()} ({genre_stats['count'].max()} titles)")
        
        # Language Analysis
        print(f"\n🌍 LANGUAGE INSIGHTS:")
        lang_stats = self.df.groupby('Language')['IMDb_Rating'].agg(['mean', 'count'])
        lang_stats = lang_stats[lang_stats['count'] >= 5]  # Filter for significance
        best_lang = lang_stats['mean'].idxmax()
        print(f"Best Performing Language: {best_lang} (Avg: {lang_stats.loc[best_lang, 'mean']:.2f})")
        print(f"Most Common Language: {self.df['Language'].mode()[0]} ({self.df['Language'].value_counts().iloc[0]} titles)")
        
        # Yearly Trends
        print(f"\n📈 YEARLY TRENDS:")
        yearly_stats = self.df.groupby('Release_Year')['IMDb_Rating'].mean()
        best_year = yearly_stats.idxmax()
        worst_year = yearly_stats.idxmin()
        print(f"Best Year for Ratings: {best_year} (Avg: {yearly_stats[best_year]:.2f})")
        print(f"Worst Year for Ratings: {worst_year} (Avg: {yearly_stats[worst_year]:.2f})")
        
        # Budget Impact
        print(f"\n💰 BUDGET IMPACT:")
        budget_stats = self.df.groupby('Budget_Category')['IMDb_Rating'].mean()
        for budget, rating in budget_stats.items():
            print(f"{budget} Budget: {rating:.2f} average rating")
        
        # Quality Distribution
        print(f"\n⭐ QUALITY DISTRIBUTION:")
        excellent = len(self.df[self.df['IMDb_Rating'] >= 8.0])
        good = len(self.df[(self.df['IMDb_Rating'] >= 7.0) & (self.df['IMDb_Rating'] < 8.0)])
        average = len(self.df[(self.df['IMDb_Rating'] >= 6.0) & (self.df['IMDb_Rating'] < 7.0)])
        below_avg = len(self.df[self.df['IMDb_Rating'] < 6.0])
        
        print(f"Excellent (8.0+): {excellent} titles ({excellent/len(self.df)*100:.1f}%)")
        print(f"Good (7.0-7.9): {good} titles ({good/len(self.df)*100:.1f}%)")
        print(f"Average (6.0-6.9): {average} titles ({average/len(self.df)*100:.1f}%)")
        print(f"Below Average (<6.0): {below_avg} titles ({below_avg/len(self.df)*100:.1f}%)")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self):
        """Run the complete initial visual analysis"""
        print("Starting Netflix Originals IMDb Analysis...")
        print("Generating Initial Visual Representation of Key Findings...\n")
        
        # Create the main dashboard
        self.create_overview_dashboard()
        
        # Generate detailed findings report
        self.create_detailed_findings_report()
        
        print("\n✅ Initial Visual Representation Complete!")
        print("Key findings have been visualized and summarized.")

# Execute the analysis
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = NetflixVisualization()
    
    # Run the complete initial visual analysis
    analyzer.run_complete_analysis()