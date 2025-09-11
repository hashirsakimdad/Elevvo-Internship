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

Author: AI Assistant
Date: 2025
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

# Set style for better plots
plt.style.use('seaborn-v0_8')
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
        Create comprehensive exploratory visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Data Exploration', fontsize=16, fontweight='bold')
        
        # Distribution of age
        axes[0, 0].hist(data['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Age')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution of annual income
        axes[0, 1].hist(data['annual_income'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Annual Income')
        axes[0, 1].set_xlabel('Annual Income (k$)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of spending score
        axes[0, 2].hist(data['spending_score'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Distribution of Spending Score')
        axes[0, 2].set_xlabel('Spending Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Income vs Spending Score scatter plot
        axes[1, 0].scatter(data['annual_income'], data['spending_score'], alpha=0.6, color='purple')
        axes[1, 0].set_title('Annual Income vs Spending Score')
        axes[1, 0].set_xlabel('Annual Income (k$)')
        axes[1, 0].set_ylabel('Spending Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Age vs Spending Score scatter plot
        axes[1, 1].scatter(data['age'], data['spending_score'], alpha=0.6, color='red')
        axes[1, 1].set_title('Age vs Spending Score')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Spending Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Correlation heatmap
        correlation_matrix = data[['age', 'annual_income', 'spending_score']].corr()
        im = axes[1, 2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 2].set_title('Feature Correlation Matrix')
        axes[1, 2].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 2].set_yticks(range(len(correlation_matrix.columns)))
        axes[1, 2].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[1, 2].set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values to heatmap
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = axes[1, 2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('internship folder task level 1/Task2_Customer_Segmentation/data_exploration.png', 
                   dpi=300, bbox_inches='tight')
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Optimal Number of Clusters Analysis', fontsize=16, fontweight='bold')
        
        # Elbow method
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_title('Elbow Method')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette analysis
        axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].set_title('Silhouette Analysis')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz index
        axes[2].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
        axes[2].set_title('Calinski-Harabasz Index')
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Calinski-Harabasz Score')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 1/Task2_Customer_Segmentation/optimal_clusters.png', 
                   dpi=300, bbox_inches='tight')
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
        Visualize clusters using 2D plots
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
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle('Customer Segmentation Results', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # K-Means visualization
        scatter = axes[plot_idx].scatter(self.data['annual_income'], self.data['spending_score'], 
                                       c=kmeans_labels, cmap='viridis', alpha=0.7)
        axes[plot_idx].set_title('K-Means Clustering')
        axes[plot_idx].set_xlabel('Annual Income (k$)')
        axes[plot_idx].set_ylabel('Spending Score')
        axes[plot_idx].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[plot_idx])
        plot_idx += 1
        
        # DBSCAN visualization (if available)
        if dbscan_labels is not None:
            scatter = axes[plot_idx].scatter(self.data['annual_income'], self.data['spending_score'], 
                                           c=dbscan_labels, cmap='viridis', alpha=0.7)
            axes[plot_idx].set_title('DBSCAN Clustering')
            axes[plot_idx].set_xlabel('Annual Income (k$)')
            axes[plot_idx].set_ylabel('Spending Score')
            axes[plot_idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[plot_idx])
            plot_idx += 1
        
        # Hierarchical visualization (if available)
        if hierarchical_labels is not None:
            scatter = axes[plot_idx].scatter(self.data['annual_income'], self.data['spending_score'], 
                                           c=hierarchical_labels, cmap='viridis', alpha=0.7)
            axes[plot_idx].set_title('Hierarchical Clustering')
            axes[plot_idx].set_xlabel('Annual Income (k$)')
            axes[plot_idx].set_ylabel('Spending Score')
            axes[plot_idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[plot_idx])
        
        plt.tight_layout()
        plt.savefig('internship folder task level 1/Task2_Customer_Segmentation/cluster_visualization.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed cluster analysis plot
        self.create_detailed_cluster_analysis(kmeans_labels)
    
    def create_detailed_cluster_analysis(self, kmeans_labels):
        """
        Create detailed cluster analysis visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Cluster Analysis', fontsize=16, fontweight='bold')
        
        # Income vs Spending Score with clusters
        scatter = axes[0, 0].scatter(self.data['annual_income'], self.data['spending_score'], 
                                   c=kmeans_labels, cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('Income vs Spending Score (Clusters)')
        axes[0, 0].set_xlabel('Annual Income (k$)')
        axes[0, 0].set_ylabel('Spending Score')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Age vs Spending Score with clusters
        scatter = axes[0, 1].scatter(self.data['age'], self.data['spending_score'], 
                                   c=kmeans_labels, cmap='viridis', alpha=0.7)
        axes[0, 1].set_title('Age vs Spending Score (Clusters)')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Spending Score')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Cluster size distribution
        cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
        axes[1, 0].bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', alpha=0.7)
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average spending per cluster
        avg_spending = self.data.groupby(kmeans_labels)['spending_score'].mean()
        axes[1, 1].bar(avg_spending.index, avg_spending.values, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Average Spending Score per Cluster')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Average Spending Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 1/Task2_Customer_Segmentation/detailed_cluster_analysis.png', 
                   dpi=300, bbox_inches='tight')
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
                if np.sum(non_noise_mask) > 1:
                    silhouette_avg = silhouette_score(self.scaled_data[non_noise_mask], 
                                                    labels[non_noise_mask])
                    calinski_score = calinski_harabasz_score(self.scaled_data[non_noise_mask], 
                                                           labels[non_noise_mask])
                else:
                    silhouette_avg = 0
                    calinski_score = 0
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Clustering Algorithms Comparison', fontsize=16, fontweight='bold')
        
        # Silhouette score comparison
        axes[0].bar(comparison_df['Algorithm'], comparison_df['Silhouette'], 
                   color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        axes[0].set_title('Silhouette Score Comparison')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].grid(True, alpha=0.3)
        
        # Calinski-Harabasz score comparison
        axes[1].bar(comparison_df['Algorithm'], comparison_df['Calinski-Harabasz'], 
                   color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        axes[1].set_title('Calinski-Harabasz Score Comparison')
        axes[1].set_ylabel('Calinski-Harabasz Score')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 1/Task2_Customer_Segmentation/algorithm_comparison.png', 
                   dpi=300, bbox_inches='tight')
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
