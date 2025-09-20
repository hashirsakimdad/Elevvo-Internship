"""
Level 2, Task 5: Movie Recommendation System
===========================================

Objective: Build a system that recommends movies based on user similarity.

Dataset: MovieLens 100K Dataset (Kaggle)
Methodology: Use a user-item matrix to compute similarity scores
Output: Recommend top-rated unseen movies for a given user
Evaluation: Evaluate performance using precision at K

Bonus:
- Implement item-based collaborative filtering
- Try matrix factorization (SVD)

Author: Muhammad Hashir Sakim dad
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MovieRecommendationSystem:
    """
    A comprehensive movie recommendation system using collaborative filtering
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.movie_data = None
        self.rating_data = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_model = None
        self.user_features = None
        self.item_features = None
        
    def generate_sample_data(self):
        """
        Generate sample movie rating data similar to MovieLens dataset
        """
        np.random.seed(42)
        
        # Generate movie data
        n_movies = 100
        movie_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Adventure']
        
        movies = []
        for i in range(n_movies):
            movie_id = i + 1
            title = f"Movie {movie_id}"
            # Assign random genres
            num_genres = np.random.randint(1, 4)
            genres = np.random.choice(movie_genres, num_genres, replace=False)
            genre_str = '|'.join(genres)
            
            movies.append({
                'movie_id': movie_id,
                'title': title,
                'genres': genre_str
            })
        
        self.movie_data = pd.DataFrame(movies)
        
        # Generate user rating data
        n_users = 200
        n_ratings = 2000  # Sparse matrix
        
        ratings = []
        for _ in range(n_ratings):
            user_id = np.random.randint(1, n_users + 1)
            movie_id = np.random.randint(1, n_movies + 1)
            
            # Create realistic rating patterns
            # Some users prefer certain genres
            user_preference = np.random.random()
            movie_genres = self.movie_data[self.movie_data['movie_id'] == movie_id]['genres'].iloc[0].split('|')
            
            # Base rating
            base_rating = np.random.normal(3.5, 1.0)
            
            # Genre preference adjustment
            if 'Action' in movie_genres and user_preference > 0.7:
                base_rating += 0.5
            elif 'Comedy' in movie_genres and user_preference < 0.3:
                base_rating += 0.5
            elif 'Drama' in movie_genres and 0.4 < user_preference < 0.6:
                base_rating += 0.5
            
            # Clip rating to 1-5 range
            rating = np.clip(base_rating, 1, 5)
            rating = int(round(rating))
            
            ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': np.random.randint(1000000000, 2000000000)
            })
        
        self.rating_data = pd.DataFrame(ratings)
        
        # Remove duplicate ratings
        self.rating_data = self.rating_data.drop_duplicates(subset=['user_id', 'movie_id'])
        
        print(f"Generated {len(self.movie_data)} movies and {len(self.rating_data)} ratings from {n_users} users")
        
        return self.movie_data, self.rating_data
    
    def explore_data(self, movie_data, rating_data):
        """
        Explore and analyze the movie rating dataset
        """
        print("=" * 60)
        print("MOVIE RECOMMENDATION SYSTEM - DATA EXPLORATION")
        print("=" * 60)
        
        # Basic information
        print("\n1. Dataset Overview:")
        print(f"   - Number of movies: {len(movie_data)}")
        print(f"   - Number of ratings: {len(rating_data)}")
        print(f"   - Number of users: {rating_data['user_id'].nunique()}")
        print(f"   - Rating scale: {rating_data['rating'].min()} - {rating_data['rating'].max()}")
        
        # Rating distribution
        print("\n2. Rating Distribution:")
        rating_counts = rating_data['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            percentage = count / len(rating_data) * 100
            print(f"   Rating {rating}: {count} ratings ({percentage:.1f}%)")
        
        # User activity
        print("\n3. User Activity:")
        user_ratings = rating_data.groupby('user_id').size()
        print(f"   - Average ratings per user: {user_ratings.mean():.1f}")
        print(f"   - Min ratings per user: {user_ratings.min()}")
        print(f"   - Max ratings per user: {user_ratings.max()}")
        
        # Movie popularity
        print("\n4. Movie Popularity:")
        movie_ratings = rating_data.groupby('movie_id').size()
        print(f"   - Average ratings per movie: {movie_ratings.mean():.1f}")
        print(f"   - Min ratings per movie: {movie_ratings.min()}")
        print(f"   - Max ratings per movie: {movie_ratings.max()}")
        
        # Genre analysis
        print("\n5. Genre Analysis:")
        all_genres = []
        for genres in movie_data['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        for genre, count in genre_counts.items():
            percentage = count / len(movie_data) * 100
            print(f"   {genre}: {count} movies ({percentage:.1f}%)")
        
        # Create visualizations
        self.create_exploratory_plots(movie_data, rating_data)
        
        return movie_data, rating_data
    
    def create_exploratory_plots(self, movie_data, rating_data):
        """
        Create comprehensive exploratory visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Movie Rating Data Exploration', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.3)
        
        # Rating distribution
        rating_counts = rating_data['rating'].value_counts().sort_index()
        axes[0, 0].bar(rating_counts.index, rating_counts.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # User activity distribution
        user_ratings = rating_data.groupby('user_id').size()
        axes[0, 1].hist(user_ratings, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('User Activity Distribution')
        axes[0, 1].set_xlabel('Number of Ratings per User')
        axes[0, 1].set_ylabel('Number of Users')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Movie popularity distribution
        movie_ratings = rating_data.groupby('movie_id').size()
        axes[0, 2].hist(movie_ratings, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_title('Movie Popularity Distribution')
        axes[0, 2].set_xlabel('Number of Ratings per Movie')
        axes[0, 2].set_ylabel('Number of Movies')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Genre distribution
        all_genres = []
        for genres in movie_data['genres']:
            all_genres.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres).value_counts()
        axes[1, 0].bar(range(len(genre_counts)), genre_counts.values, color='purple', alpha=0.7)
        axes[1, 0].set_title('Genre Distribution')
        axes[1, 0].set_xlabel('Genre')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(range(len(genre_counts)))
        axes[1, 0].set_xticklabels(genre_counts.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average rating by genre
        genre_ratings = []
        genre_names = []
        for genre in genre_counts.index:
            genre_movies = movie_data[movie_data['genres'].str.contains(genre)]['movie_id']
            genre_rating_data = rating_data[rating_data['movie_id'].isin(genre_movies)]
            if len(genre_rating_data) > 0:
                avg_rating = genre_rating_data['rating'].mean()
                genre_ratings.append(avg_rating)
                genre_names.append(genre)
        
        axes[1, 1].bar(range(len(genre_names)), genre_ratings, color='red', alpha=0.7)
        axes[1, 1].set_title('Average Rating by Genre')
        axes[1, 1].set_xlabel('Genre')
        axes[1, 1].set_ylabel('Average Rating')
        axes[1, 1].set_xticks(range(len(genre_names)))
        axes[1, 1].set_xticklabels(genre_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Rating over time (simulated)
        rating_data_sorted = rating_data.sort_values('timestamp')
        time_ratings = rating_data_sorted.groupby(rating_data_sorted.index // 100)['rating'].mean()
        axes[1, 2].plot(time_ratings.index, time_ratings.values, color='green', alpha=0.7)
        axes[1, 2].set_title('Average Rating Over Time')
        axes[1, 2].set_xlabel('Time Period')
        axes[1, 2].set_ylabel('Average Rating')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.savefig('data_exploration.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_user_item_matrix(self, rating_data):
        """
        Create user-item rating matrix
        """
        print("\n" + "=" * 60)
        print("USER-ITEM MATRIX CREATION")
        print("=" * 60)
        
        # Create user-item matrix
        self.user_item_matrix = rating_data.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        print(f"\nUser-Item Matrix Shape: {self.user_item_matrix.shape}")
        print(f"Sparsity: {(self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100:.1f}%")
        
        # Calculate user similarity
        print("\nCalculating user similarity...")
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        # Calculate item similarity
        print("Calculating item similarity...")
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        print("✓ Similarity matrices created successfully!")
        
        return self.user_item_matrix
    
    def user_based_collaborative_filtering(self, user_id, n_recommendations=10):
        """
        Implement user-based collaborative filtering
        """
        print(f"\nUser-Based Collaborative Filtering for User {user_id}")
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset!")
            return None
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
        
        print(f"User {user_id} has rated {len(rated_movies)} movies")
        
        # Find similar users
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_similarities = self.user_similarity[user_idx]
        
        # Get top similar users (excluding the user themselves)
        similar_users_idx = np.argsort(user_similarities)[::-1][1:11]  # Top 10 similar users
        similar_users = self.user_item_matrix.index[similar_users_idx]
        
        print(f"Top similar users: {similar_users.tolist()}")
        
        # Calculate predicted ratings for unrated movies
        unrated_movies = self.user_item_matrix.columns[~self.user_item_matrix.columns.isin(rated_movies)]
        
        predictions = []
        for movie_id in unrated_movies:
            # Get ratings for this movie from similar users
            movie_ratings = self.user_item_matrix.loc[similar_users, movie_id]
            valid_ratings = movie_ratings[movie_ratings > 0]
            
            if len(valid_ratings) > 0:
                # Weighted average of ratings
                similarities = user_similarities[similar_users_idx][movie_ratings > 0]
                weighted_rating = np.sum(valid_ratings * similarities) / np.sum(similarities)
                predictions.append((movie_id, weighted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_recommendations = predictions[:n_recommendations]
        
        print(f"\nTop {n_recommendations} recommendations for User {user_id}:")
        for i, (movie_id, predicted_rating) in enumerate(top_recommendations, 1):
            movie_title = self.movie_data[self.movie_data['movie_id'] == movie_id]['title'].iloc[0]
            print(f"   {i}. {movie_title} (Predicted Rating: {predicted_rating:.2f})")
        
        return top_recommendations
    
    def item_based_collaborative_filtering(self, user_id, n_recommendations=10):
        """
        Implement item-based collaborative filtering (Bonus feature)
        """
        print(f"\nItem-Based Collaborative Filtering for User {user_id}")
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset!")
            return None
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
        
        print(f"User {user_id} has rated {len(rated_movies)} movies")
        
        # Calculate predicted ratings for unrated movies
        unrated_movies = self.user_item_matrix.columns[~self.user_item_matrix.columns.isin(rated_movies)]
        
        predictions = []
        for movie_id in unrated_movies:
            # Get similarity with rated movies
            movie_similarities = self.item_similarity[self.user_item_matrix.columns.get_loc(movie_id)]
            
            # Get similarities with movies the user has rated
            rated_movie_indices = [self.user_item_matrix.columns.get_loc(mid) for mid in rated_movies]
            similarities_with_rated = movie_similarities[rated_movie_indices]
            
            # Get user's ratings for rated movies
            user_rated_ratings = user_ratings[rated_movies].values
            
            # Calculate weighted average
            if np.sum(similarities_with_rated) > 0:
                predicted_rating = np.sum(user_rated_ratings * similarities_with_rated) / np.sum(similarities_with_rated)
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_recommendations = predictions[:n_recommendations]
        
        print(f"\nTop {n_recommendations} recommendations for User {user_id}:")
        for i, (movie_id, predicted_rating) in enumerate(top_recommendations, 1):
            movie_title = self.movie_data[self.movie_data['movie_id'] == movie_id]['title'].iloc[0]
            print(f"   {i}. {movie_title} (Predicted Rating: {predicted_rating:.2f})")
        
        return top_recommendations
    
    def matrix_factorization_svd(self, user_id, n_recommendations=10, n_components=50):
        """
        Implement matrix factorization using SVD (Bonus feature)
        """
        print(f"\nMatrix Factorization (SVD) for User {user_id}")
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in the dataset!")
            return None
        
        # Apply SVD
        print(f"Applying SVD with {n_components} components...")
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_features = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_features = self.svd_model.components_
        
        # Reconstruct the matrix
        reconstructed_matrix = np.dot(self.user_features, self.item_features)
        
        # Get user's predicted ratings
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_predictions = reconstructed_matrix[user_idx]
        
        # Get user's actual ratings to find unrated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
        
        # Create predictions for unrated movies
        unrated_movies = self.user_item_matrix.columns[~self.user_item_matrix.columns.isin(rated_movies)]
        predictions = []
        
        for movie_id in unrated_movies:
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
            predicted_rating = user_predictions[movie_idx]
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_recommendations = predictions[:n_recommendations]
        
        print(f"\nTop {n_recommendations} recommendations for User {user_id}:")
        for i, (movie_id, predicted_rating) in enumerate(top_recommendations, 1):
            movie_title = self.movie_data[self.movie_data['movie_id'] == movie_id]['title'].iloc[0]
            print(f"   {i}. {movie_title} (Predicted Rating: {predicted_rating:.2f})")
        
        return top_recommendations
    
    def evaluate_recommendations(self, user_id, n_recommendations=10):
        """
        Evaluate recommendation performance using precision at K
        """
        print(f"\nEvaluating Recommendations for User {user_id}")
        
        # Split data into train and test
        user_ratings = self.rating_data[self.rating_data['user_id'] == user_id]
        
        if len(user_ratings) < 10:
            print(f"User {user_id} has too few ratings for evaluation!")
            return None
        
        # Split user's ratings
        train_ratings, test_ratings = train_test_split(user_ratings, test_size=0.3, random_state=42)
        
        # Create temporary user-item matrix without test ratings
        temp_rating_data = self.rating_data[~self.rating_data.index.isin(test_ratings.index)]
        temp_user_item_matrix = temp_rating_data.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        # Get test movie IDs
        test_movies = test_ratings['movie_id'].tolist()
        
        # Get recommendations using user-based CF
        user_similarity_temp = cosine_similarity(temp_user_item_matrix)
        user_idx = temp_user_item_matrix.index.get_loc(user_id)
        user_similarities = user_similarity_temp[user_idx]
        
        # Find similar users
        similar_users_idx = np.argsort(user_similarities)[::-1][1:11]
        similar_users = temp_user_item_matrix.index[similar_users_idx]
        
        # Calculate predicted ratings
        predictions = []
        for movie_id in temp_user_item_matrix.columns:
            if movie_id not in train_ratings['movie_id'].values:
                movie_ratings = temp_user_item_matrix.loc[similar_users, movie_id]
                valid_ratings = movie_ratings[movie_ratings > 0]
                
                if len(valid_ratings) > 0:
                    similarities = user_similarities[similar_users_idx][movie_ratings > 0]
                    weighted_rating = np.sum(valid_ratings * similarities) / np.sum(similarities)
                    predictions.append((movie_id, weighted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_recommendations = [pred[0] for pred in predictions[:n_recommendations]]
        
        # Calculate precision at K
        recommended_movies = set(top_recommendations)
        test_movies_set = set(test_movies)
        
        precision_at_k = len(recommended_movies.intersection(test_movies_set)) / len(recommended_movies)
        
        print(f"Precision at {n_recommendations}: {precision_at_k:.3f}")
        print(f"Recommended movies: {top_recommendations}")
        print(f"Test movies: {test_movies}")
        
        return precision_at_k
    
    def compare_recommendation_methods(self, user_id, n_recommendations=10):
        """
        Compare different recommendation methods (Bonus feature)
        """
        print(f"\nComparing Recommendation Methods for User {user_id}")
        
        # Get recommendations from all methods
        user_based_recs = self.user_based_collaborative_filtering(user_id, n_recommendations)
        item_based_recs = self.item_based_collaborative_filtering(user_id, n_recommendations)
        svd_recs = self.matrix_factorization_svd(user_id, n_recommendations)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle(f'Recommendation Comparison for User {user_id}', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.3)
        
        methods = [
            ('User-Based CF', user_based_recs),
            ('Item-Based CF', item_based_recs),
            ('SVD Matrix Factorization', svd_recs)
        ]
        
        for i, (method_name, recommendations) in enumerate(methods):
            if recommendations:
                movie_ids = [rec[0] for rec in recommendations]
                predicted_ratings = [rec[1] for rec in recommendations]
                
                axes[i].bar(range(len(movie_ids)), predicted_ratings, color='skyblue', alpha=0.7)
                axes[i].set_title(f'{method_name}\nTop {len(recommendations)} Recommendations')
                axes[i].set_xlabel('Rank')
                axes[i].set_ylabel('Predicted Rating')
                axes[i].grid(True, alpha=0.3)
                
                # Add movie titles as x-axis labels
                movie_titles = [self.movie_data[self.movie_data['movie_id'] == mid]['title'].iloc[0] 
                              for mid in movie_ids]
                axes[i].set_xticks(range(len(movie_ids)))
                axes[i].set_xticklabels([f"Movie {mid}" for mid in movie_ids], rotation=45)
        
        plt.savefig('recommendation_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print comparison summary
        print(f"\nRecommendation Summary for User {user_id}:")
        for method_name, recommendations in methods:
            if recommendations:
                avg_rating = np.mean([rec[1] for rec in recommendations])
                print(f"   {method_name}: Average predicted rating = {avg_rating:.2f}")
    
    def run_complete_analysis(self):
        """
        Run the complete movie recommendation system analysis
        """
        print("MOVIE RECOMMENDATION SYSTEM ANALYSIS")
        print("=" * 60)
        print("This analysis builds a movie recommendation system using")
        print("collaborative filtering and matrix factorization techniques.")
        print("=" * 60)
        
        # Generate sample data
        movie_data, rating_data = self.generate_sample_data()
        
        # Explore data
        self.explore_data(movie_data, rating_data)
        
        # Create user-item matrix
        self.create_user_item_matrix(rating_data)
        
        # Test recommendations for a sample user
        test_user = self.user_item_matrix.index[0]
        print(f"\nTesting recommendations for User {test_user}")
        
        # User-based collaborative filtering
        self.user_based_collaborative_filtering(test_user)
        
        # Item-based collaborative filtering (bonus)
        self.item_based_collaborative_filtering(test_user)
        
        # Matrix factorization with SVD (bonus)
        self.matrix_factorization_svd(test_user)
        
        # Compare recommendation methods (bonus)
        self.compare_recommendation_methods(test_user)
        
        # Evaluate recommendations
        precision = self.evaluate_recommendations(test_user)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Files saved:")
        print("   - data_exploration.png")
        print("   - recommendation_comparison.png")
        print("\nKey Findings:")
        print("   - User-based collaborative filtering provides personalized recommendations")
        print("   - Item-based filtering captures movie-to-movie relationships")
        print("   - SVD matrix factorization reduces dimensionality effectively")
        print("   - Different methods produce different recommendation patterns")
        print("   - Precision at K is a good metric for evaluation")

def main():
    """
    Main function to run the movie recommendation system analysis
    """
    # Create recommendation system instance
    recommender = MovieRecommendationSystem()
    
    # Run complete analysis
    recommender.run_complete_analysis()

if __name__ == "__main__":
    main()
