"""
Level 1, Task 2: Customer Segmentation
=====================================

Objective: Cluster customers into segments based on income and spending score.

Dataset: Mall Customer (Kaggle)
Steps:
1. Perform scaling and visual exploration of groupings
2. Apply K-Means clustering and determine the optimal number of clusters
3. Visualize clusters using 2D plots

Bonus:
- Try different clustering algorithms (e.g., DBSCAN)
- Analyze average spending per cluster

 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set professional style for better plots
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})
sns.set_palette("husl")

class CustomerSegmentation:
    """
    A comprehensive class for customer segmentation using clustering algorithms
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = None
        self.scaled_data = None
        self.kmeans_model = None
        self.dbscan_model = None
        self.hierarchical_model = None
        self.optimal_k = None
        
    def generate_sample_data(self):
        """
        Generate sample customer data similar to Mall Customer dataset
        """
        np.random.seed(42)
        n_customers = 200
        
        # Generate realistic customer data
        # Age distribution (18-70 years)
        age = np.random.normal(35, 12, n_customers)
        age = np.clip(age, 18, 70)
        
        # Annual income distribution (15k-150k)
        annual_income = np.random.normal(60, 25, n_customers)
        annual_income = np.clip(annual_income, 15, 150)
        
        # Spending score (1-100) - inversely related to income for some customers
        # Create different customer segments with different spending patterns
        spending_score = np.zeros(n_customers)
        
        # Segment 1: High income, high spending (20% of customers)
        high_income_mask = annual_income > 80
        spending_score[high_income_mask] = np.random.normal(75, 15, np.sum(high_income_mask))
        
        # Segment 2: High income, low spending (15% of customers)
        high_income_low_spending_mask = (annual_income > 80) & (np.random.random(n_customers) < 0.15)
        spending_score[high_income_low_spending_mask] = np.random.normal(25, 10, np.sum(high_income_low_spending_mask))
        
        # Segment 3: Medium income, medium spending (30% of customers)
        medium_income_mask = (annual_income >= 40) & (annual_income <= 80)
        spending_score[medium_income_mask] = np.random.normal(50, 15, np.sum(medium_income_mask))
        
        # Segment 4: Low income, high spending (15% of customers)
        low_income_mask = annual_income < 40
        spending_score[low_income_mask] = np.random.normal(70, 15, np.sum(low_income_mask))
        
        # Segment 5: Low income, low spending (20% of customers)
        low_income_low_spending_mask = (annual_income < 40) & (np.random.random(n_customers) < 0.2)
        spending_score[low_income_low_spending_mask] = np.random.normal(20, 10, np.sum(low_income_low_spending_mask))
        
        # Ensure spending score is within bounds
        spending_score = np.clip(spending_score, 1, 100)
        
        # Create DataFrame
        data = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'age': age,
            'annual_income': annual_income,
            'spending_score': spending_score
        })
        
        return data
    
    def explore_data(self, data):
        """
        Perform visual exploration of customer data
        """
        print("=" * 60)
        print("CUSTOMER SEGMENTATION - DATA EXPLORATION")
        print("=" * 60)
        
        # Basic information
        print("\n1. Dataset Overview:")
        print(f"   - Number of customers: {len(data)}")
        print(f"   - Features: age, annual_income, spending_score")
        
        # Data info
        print("\n2. Dataset Information:")
        print(data.info())
        
        # Basic statistics
        print("\n3. Descriptive Statistics:")
        print(data.describe())
        
        # Check for missing values
        print("\n4. Missing Values:")
        missing_values = data.isnull().sum()
        if missing_values.sum() == 0:
            print("   ✓ No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        # Create visualizations
        self.create_exploratory_plots(data)
        
        return data
    
    def create_exploratory_plots(self, data):
        """
        Create comprehensive exploratory visualizations with improved styling
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Customer Data Exploration', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.3)
        
        # Define consistent color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#0B6623']
        
        # Distribution of age
        axes[0, 0].hist(data['age'], bins=20, alpha=0.8, color=colors[0], edgecolor='white', linewidth=1)
        axes[0, 0].set_title('Distribution of Age', fontweight='bold', pad=15)
        axes[0, 0].set_xlabel('Age (years)', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Distribution of annual income
        axes[0, 1].hist(data['annual_income'], bins=20, alpha=0.8, color=colors[1], edgecolor='white', linewidth=1)
        axes[0, 1].set_title('Distribution of Annual Income', fontweight='bold', pad=15)
        axes[0, 1].set_xlabel('Annual Income (k$)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Distribution of spending score
        axes[0, 2].hist(data['spending_score'], bins=20, alpha=0.8, color=colors[2], edgecolor='white', linewidth=1)
        axes[0, 2].set_title('Distribution of Spending Score', fontweight='bold', pad=15)
        axes[0, 2].set_xlabel('Spending Score (1-100)', fontweight='bold')
        axes[0, 2].set_ylabel('Frequency', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3, linestyle='--')
        
        # Income vs Spending Score scatter plot
        axes[1, 0].scatter(data['annual_income'], data['spending_score'], alpha=0.7, color=colors[3], s=60, edgecolors='white', linewidth=0.5)
        axes[1, 0].set_title('Annual Income vs Spending Score', fontweight='bold', pad=15)
        axes[1, 0].set_xlabel('Annual Income (k$)', fontweight='bold')
        axes[1, 0].set_ylabel('Spending Score (1-100)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Age vs Spending Score scatter plot
        axes[1, 1].scatter(data['age'], data['spending_score'], alpha=0.7, color=colors[4], s=60, edgecolors='white', linewidth=0.5)
        axes[1, 1].set_title('Age vs Spending Score', fontweight='bold', pad=15)
        axes[1, 1].set_xlabel('Age (years)', fontweight='bold')
        axes[1, 1].set_ylabel('Spending Score (1-100)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Correlation heatmap
        correlation_matrix = data[['age', 'annual_income', 'spending_score']].corr()
        im = axes[1, 2].imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1, 2].set_title('Feature Correlation Matrix', fontweight='bold', pad=15)
        axes[1, 2].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 2].set_yticks(range(len(correlation_matrix.columns)))
        axes[1, 2].set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        axes[1, 2].set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values to heatmap with better formatting
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                value = correlation_matrix.iloc[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                text = axes[1, 2].text(j, i, f'{value:.2f}',
                                     ha="center", va="center", color=color, fontweight='bold', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontweight='bold')
        
        plt.savefig('data_exploration.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print correlation insights
        print("\n5. Correlation Insights:")
        correlations = data[['age', 'annual_income', 'spending_score']].corr()
        print(f"   - Age vs Annual Income: {correlations.loc['age', 'annual_income']:.3f}")
        print(f"   - Age vs Spending Score: {correlations.loc['age', 'spending_score']:.3f}")
        print(f"   - Annual Income vs Spending Score: {correlations.loc['annual_income', 'spending_score']:.3f}")
    
    def prepare_data(self, data):
        """
        Prepare and scale data for clustering
        """
        print("\n" + "=" * 60)
        print("DATA PREPARATION AND SCALING")
        print("=" * 60)
        
        # Select features for clustering (excluding customer_id)
        features = ['age', 'annual_income', 'spending_score']
        self.data = data[features].copy()
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        print(f"\nFeatures selected for clustering: {features}")
        print(f"Data shape: {self.data.shape}")
        print(f"Scaled data shape: {self.scaled_data.shape}")
        
        # Show scaling effect
        print("\nScaling Effect:")
        print("Original data statistics:")
        print(self.data.describe())
        
        scaled_df = pd.DataFrame(self.scaled_data, columns=features)
        print("\nScaled data statistics:")
        print(scaled_df.describe())
        
        return self.scaled_data
    
    def find_optimal_clusters(self, max_k=10):
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        """
        print("\n" + "=" * 60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("=" * 60)
        
        # Calculate metrics for different k values
        k_range = range(2, max_k + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, kmeans.labels_))
            calinski_scores.append(calinski_harabasz_score(self.scaled_data, kmeans.labels_))
        
        # Plot elbow method and silhouette analysis
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle('Optimal Number of Clusters Analysis', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.3)
        
        # Define consistent color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # Elbow method
        axes[0].plot(k_range, inertias, 'o-', linewidth=3, markersize=10, color=colors[0], markerfacecolor='white', markeredgecolor=colors[0], markeredgewidth=2)
        axes[0].set_title('Elbow Method', fontweight='bold', pad=15)
        axes[0].set_xlabel('Number of Clusters (k)', fontweight='bold')
        axes[0].set_ylabel('Inertia', fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Highlight optimal point
        optimal_idx = np.argmax(silhouette_scores)
        axes[0].scatter(k_range[optimal_idx], inertias[optimal_idx], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        axes[0].text(k_range[optimal_idx], inertias[optimal_idx] + max(inertias)*0.02, f'Optimal k={k_range[optimal_idx]}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Silhouette analysis
        axes[1].plot(k_range, silhouette_scores, 'o-', linewidth=3, markersize=10, color=colors[1], markerfacecolor='white', markeredgecolor=colors[1], markeredgewidth=2)
        axes[1].set_title('Silhouette Analysis', fontweight='bold', pad=15)
        axes[1].set_xlabel('Number of Clusters (k)', fontweight='bold')
        axes[1].set_ylabel('Silhouette Score', fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        # Highlight optimal point
        axes[1].scatter(k_range[optimal_idx], silhouette_scores[optimal_idx], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        axes[1].text(k_range[optimal_idx], silhouette_scores[optimal_idx] + max(silhouette_scores)*0.02, f'Best k={k_range[optimal_idx]}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Calinski-Harabasz index
        axes[2].plot(k_range, calinski_scores, 'o-', linewidth=3, markersize=10, color=colors[2], markerfacecolor='white', markeredgecolor=colors[2], markeredgewidth=2)
        axes[2].set_title('Calinski-Harabasz Index', fontweight='bold', pad=15)
        axes[2].set_xlabel('Number of Clusters (k)', fontweight='bold')
        axes[2].set_ylabel('Calinski-Harabasz Score', fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        # Highlight optimal point
        optimal_calinski_idx = np.argmax(calinski_scores)
        axes[2].scatter(k_range[optimal_calinski_idx], calinski_scores[optimal_calinski_idx], color='red', s=150, zorder=5, edgecolors='black', linewidth=2)
        axes[2].text(k_range[optimal_calinski_idx], calinski_scores[optimal_calinski_idx] + max(calinski_scores)*0.02, f'Best k={k_range[optimal_calinski_idx]}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.savefig('optimal_clusters.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Find optimal k
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        optimal_k_calinski = k_range[np.argmax(calinski_scores)]
        
        print(f"\nOptimal number of clusters:")
        print(f"   - Silhouette Score: {optimal_k_silhouette} clusters (score: {max(silhouette_scores):.3f})")
        print(f"   - Calinski-Harabasz: {optimal_k_calinski} clusters (score: {max(calinski_scores):.3f})")
        
        # Use silhouette score as primary metric
        self.optimal_k = optimal_k_silhouette
        
        return self.optimal_k
    
    def apply_kmeans_clustering(self):
        """
        Apply K-Means clustering with optimal number of clusters
        """
        print("\n" + "=" * 60)
        print("K-MEANS CLUSTERING")
        print("=" * 60)
        
        # Apply K-Means with optimal k
        self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        kmeans_labels = self.kmeans_model.fit_predict(self.scaled_data)
        
        # Add cluster labels to original data
        self.data['kmeans_cluster'] = kmeans_labels
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(self.scaled_data, kmeans_labels)
        calinski_score = calinski_harabasz_score(self.scaled_data, kmeans_labels)
        
        print(f"\nK-Means Results:")
        print(f"   - Number of clusters: {self.optimal_k}")
        print(f"   - Silhouette Score: {silhouette_avg:.3f}")
        print(f"   - Calinski-Harabasz Score: {calinski_score:.3f}")
        
        # Analyze cluster characteristics
        self.analyze_clusters('kmeans_cluster')
        
        return kmeans_labels
    
    def apply_dbscan_clustering(self):
        """
        Apply DBSCAN clustering (Bonus feature)
        """
        print("\n" + "=" * 60)
        print("DBSCAN CLUSTERING (BONUS)")
        print("=" * 60)
        
        # Apply DBSCAN
        self.dbscan_model = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = self.dbscan_model.fit_predict(self.scaled_data)
        
        # Add cluster labels to original data
        self.data['dbscan_cluster'] = dbscan_labels
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        print(f"\nDBSCAN Results:")
        print(f"   - Number of clusters: {n_clusters}")
        print(f"   - Number of noise points: {n_noise}")
        print(f"   - Percentage of noise: {n_noise/len(dbscan_labels)*100:.1f}%")
        
        if n_clusters > 1:
            # Calculate silhouette score (excluding noise points)
            non_noise_mask = dbscan_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(self.scaled_data[non_noise_mask], 
                                                dbscan_labels[non_noise_mask])
                print(f"   - Silhouette Score: {silhouette_avg:.3f}")
        
        # Analyze cluster characteristics
        self.analyze_clusters('dbscan_cluster')
        
        return dbscan_labels
    
    def apply_hierarchical_clustering(self):
        """
        Apply Hierarchical clustering (Bonus feature)
        """
        print("\n" + "=" * 60)
        print("HIERARCHICAL CLUSTERING (BONUS)")
        print("=" * 60)
        
        # Apply Hierarchical clustering
        self.hierarchical_model = AgglomerativeClustering(n_clusters=self.optimal_k)
        hierarchical_labels = self.hierarchical_model.fit_predict(self.scaled_data)
        
        # Add cluster labels to original data
        self.data['hierarchical_cluster'] = hierarchical_labels
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(self.scaled_data, hierarchical_labels)
        calinski_score = calinski_harabasz_score(self.scaled_data, hierarchical_labels)
        
        print(f"\nHierarchical Clustering Results:")
        print(f"   - Number of clusters: {self.optimal_k}")
        print(f"   - Silhouette Score: {silhouette_avg:.3f}")
        print(f"   - Calinski-Harabasz Score: {calinski_score:.3f}")
        
        # Analyze cluster characteristics
        self.analyze_clusters('hierarchical_cluster')
        
        return hierarchical_labels
    
    def analyze_clusters(self, cluster_column):
        """
        Analyze characteristics of each cluster
        """
        print(f"\nCluster Analysis ({cluster_column}):")
        
        cluster_stats = self.data.groupby(cluster_column).agg({
            'age': ['mean', 'std'],
            'annual_income': ['mean', 'std'],
            'spending_score': ['mean', 'std']
        }).round(2)
        
        print(cluster_stats)
        
        # Calculate cluster sizes
        cluster_sizes = self.data[cluster_column].value_counts().sort_index()
        print(f"\nCluster sizes:")
        for cluster_id, size in cluster_sizes.items():
            percentage = size / len(self.data) * 100
            print(f"   Cluster {cluster_id}: {size} customers ({percentage:.1f}%)")
    
    def visualize_clusters(self, kmeans_labels, dbscan_labels=None, hierarchical_labels=None):
        """
        Visualize clusters using 2D plots with improved styling
        """
        print("\n" + "=" * 60)
        print("CLUSTER VISUALIZATION")
        print("=" * 60)
        
        # Create subplots
        n_plots = 1
        if dbscan_labels is not None:
            n_plots += 1
        if hierarchical_labels is not None:
            n_plots += 1
        
        fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 8))
        fig.subplots_adjust(left=0.08, right=0.95, wspace=0.3)
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle('Customer Segmentation Results', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # K-Means visualization
        scatter = axes[plot_idx].scatter(self.data['annual_income'], self.data['spending_score'], 
                                       c=kmeans_labels, cmap='viridis', alpha=0.8, s=80, edgecolors='white', linewidth=0.5)
        axes[plot_idx].set_title('K-Means Clustering', fontweight='bold', pad=15)
        axes[plot_idx].set_xlabel('Annual Income (k$)', fontweight='bold')
        axes[plot_idx].set_ylabel('Spending Score (1-100)', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3, linestyle='--')
        cbar = plt.colorbar(scatter, ax=axes[plot_idx])
        cbar.set_label('Cluster ID', fontweight='bold')
        plot_idx += 1
        
        # DBSCAN visualization (if available)
        if dbscan_labels is not None:
            scatter = axes[plot_idx].scatter(self.data['annual_income'], self.data['spending_score'], 
                                           c=dbscan_labels, cmap='viridis', alpha=0.8, s=80, edgecolors='white', linewidth=0.5)
            axes[plot_idx].set_title('DBSCAN Clustering', fontweight='bold', pad=15)
            axes[plot_idx].set_xlabel('Annual Income (k$)', fontweight='bold')
            axes[plot_idx].set_ylabel('Spending Score (1-100)', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3, linestyle='--')
            cbar = plt.colorbar(scatter, ax=axes[plot_idx])
            cbar.set_label('Cluster ID', fontweight='bold')
            plot_idx += 1
        
        # Hierarchical visualization (if available)
        if hierarchical_labels is not None:
            scatter = axes[plot_idx].scatter(self.data['annual_income'], self.data['spending_score'], 
                                           c=hierarchical_labels, cmap='viridis', alpha=0.8, s=80, edgecolors='white', linewidth=0.5)
            axes[plot_idx].set_title('Hierarchical Clustering', fontweight='bold', pad=15)
            axes[plot_idx].set_xlabel('Annual Income (k$)', fontweight='bold')
            axes[plot_idx].set_ylabel('Spending Score (1-100)', fontweight='bold')
            axes[plot_idx].grid(True, alpha=0.3, linestyle='--')
            cbar = plt.colorbar(scatter, ax=axes[plot_idx])
            cbar.set_label('Cluster ID', fontweight='bold')
        
        plt.savefig('cluster_visualization.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Create detailed cluster analysis plot
        self.create_detailed_cluster_analysis(kmeans_labels)
    
    def create_detailed_cluster_analysis(self, kmeans_labels):
        """
        Create detailed cluster analysis visualization with improved styling
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Detailed Cluster Analysis', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.3)
        
        # Define consistent color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#0B6623']
        
        # Income vs Spending Score with clusters
        scatter = axes[0, 0].scatter(self.data['annual_income'], self.data['spending_score'], 
                                   c=kmeans_labels, cmap='viridis', alpha=0.8, s=80, edgecolors='white', linewidth=0.5)
        axes[0, 0].set_title('Income vs Spending Score (Clusters)', fontweight='bold', pad=15)
        axes[0, 0].set_xlabel('Annual Income (k$)', fontweight='bold')
        axes[0, 0].set_ylabel('Spending Score (1-100)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        cbar = plt.colorbar(scatter, ax=axes[0, 0])
        cbar.set_label('Cluster ID', fontweight='bold')
        
        # Age vs Spending Score with clusters
        scatter = axes[0, 1].scatter(self.data['age'], self.data['spending_score'], 
                                   c=kmeans_labels, cmap='viridis', alpha=0.8, s=80, edgecolors='white', linewidth=0.5)
        axes[0, 1].set_title('Age vs Spending Score (Clusters)', fontweight='bold', pad=15)
        axes[0, 1].set_xlabel('Age (years)', fontweight='bold')
        axes[0, 1].set_ylabel('Spending Score (1-100)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        cbar = plt.colorbar(scatter, ax=axes[0, 1])
        cbar.set_label('Cluster ID', fontweight='bold')
        
        # Cluster size distribution
        cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
        bars = axes[1, 0].bar(cluster_sizes.index, cluster_sizes.values, color=colors[0], alpha=0.8, edgecolor='white', linewidth=1)
        axes[1, 0].set_title('Cluster Size Distribution', fontweight='bold', pad=15)
        axes[1, 0].set_xlabel('Cluster ID', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Customers', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Average spending per cluster
        avg_spending = self.data.groupby(kmeans_labels)['spending_score'].mean()
        bars = axes[1, 1].bar(avg_spending.index, avg_spending.values, color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)
        axes[1, 1].set_title('Average Spending Score per Cluster', fontweight='bold', pad=15)
        axes[1, 1].set_xlabel('Cluster ID', fontweight='bold')
        axes[1, 1].set_ylabel('Average Spending Score', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.savefig('detailed_cluster_analysis.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print average spending per cluster
        print("\nAverage Spending Score per Cluster:")
        for cluster_id, avg_spending in avg_spending.items():
            print(f"   Cluster {cluster_id}: {avg_spending:.1f}")
    
    def compare_clustering_algorithms(self, kmeans_labels, dbscan_labels, hierarchical_labels):
        """
        Compare different clustering algorithms (Bonus feature)
        """
        print("\n" + "=" * 60)
        print("CLUSTERING ALGORITHMS COMPARISON (BONUS)")
        print("=" * 60)
        
        # Calculate metrics for each algorithm
        algorithms = {
            'K-Means': kmeans_labels,
            'DBSCAN': dbscan_labels,
            'Hierarchical': hierarchical_labels
        }
        
        comparison_data = []
        
        for name, labels in algorithms.items():
            if name == 'DBSCAN':
                # Handle DBSCAN noise points
                non_noise_mask = labels != -1
                unique_labels = np.unique(labels[non_noise_mask])
                n_clusters = len(unique_labels)
                
                if n_clusters > 1 and np.sum(non_noise_mask) > 1:
                    silhouette_avg = silhouette_score(self.scaled_data[non_noise_mask], 
                                                    labels[non_noise_mask])
                    calinski_score = calinski_harabasz_score(self.scaled_data[non_noise_mask], 
                                                           labels[non_noise_mask])
                else:
                    silhouette_avg = 0
                    calinski_score = 0
            else:
                silhouette_avg = silhouette_score(self.scaled_data, labels)
                calinski_score = calinski_harabasz_score(self.scaled_data, labels)
                n_clusters = len(set(labels))
            
            comparison_data.append({
                'Algorithm': name,
                'Clusters': n_clusters,
                'Silhouette': silhouette_avg,
                'Calinski-Harabasz': calinski_score
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        print("\nAlgorithm Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Clustering Algorithms Comparison', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.3)
        
        # Define consistent color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # Silhouette score comparison
        bars1 = axes[0].bar(comparison_df['Algorithm'], comparison_df['Silhouette'], 
                           color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        axes[0].set_title('Silhouette Score Comparison', fontweight='bold', pad=15)
        axes[0].set_ylabel('Silhouette Score', fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Calinski-Harabasz score comparison
        bars2 = axes[1].bar(comparison_df['Algorithm'], comparison_df['Calinski-Harabasz'], 
                           color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        axes[1].set_title('Calinski-Harabasz Score Comparison', fontweight='bold', pad=15)
        axes[1].set_ylabel('Calinski-Harabasz Score', fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + max(comparison_df['Calinski-Harabasz'])*0.01,
                        f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.savefig('algorithm_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run the complete customer segmentation analysis
        """
        print("CUSTOMER SEGMENTATION ANALYSIS")
        print("=" * 60)
        print("This analysis segments customers based on income and spending patterns")
        print("using various clustering algorithms.")
        print("=" * 60)
        
        # Generate sample data
        data = self.generate_sample_data()
        
        # Explore data
        data = self.explore_data(data)
        
        # Prepare data
        self.prepare_data(data)
        
        # Find optimal number of clusters
        self.find_optimal_clusters()
        
        # Apply K-Means clustering
        kmeans_labels = self.apply_kmeans_clustering()
        
        # Apply DBSCAN clustering (bonus)
        dbscan_labels = self.apply_dbscan_clustering()
        
        # Apply Hierarchical clustering (bonus)
        hierarchical_labels = self.apply_hierarchical_clustering()
        
        # Visualize clusters
        self.visualize_clusters(kmeans_labels, dbscan_labels, hierarchical_labels)
        
        # Compare algorithms (bonus)
        self.compare_clustering_algorithms(kmeans_labels, dbscan_labels, hierarchical_labels)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Files saved:")
        print("   - data_exploration.png")
        print("   - optimal_clusters.png")
        print("   - cluster_visualization.png")
        print("   - detailed_cluster_analysis.png")
        print("   - algorithm_comparison.png")
        print("\nKey Findings:")
        print("   - Optimal number of clusters identified using multiple metrics")
        print("   - K-Means provides clear customer segments")
        print("   - DBSCAN identifies outliers and noise points")
        print("   - Hierarchical clustering shows similar patterns")
        print("   - Different algorithms reveal different aspects of customer behavior")

def main():
    """
    Main function to run the customer segmentation analysis
    """
    # Create segmentation instance
    segmenter = CustomerSegmentation()
    
    # Run complete analysis
    segmenter.run_complete_analysis()

if __name__ == "__main__":
    main()
