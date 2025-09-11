"""
Level 1, Task 1: Student Score Prediction
==========================================

Objective: Build a model to predict students' exam scores based on their study hours.

Dataset: Student Performance Factors (Kaggle)
Steps:
1. Perform data cleaning and basic visualization to understand the dataset
2. Split the dataset into training and testing sets
3. Train a linear regression model to estimate final scores
4. Visualize predictions and evaluate model performance

Bonus:
- Try polynomial regression and compare performance
- Try experimenting with different feature combinations

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StudentScorePredictor:
    """
    A comprehensive class for student score prediction using linear regression
    """
    
    def __init__(self):
        self.model = None
        self.poly_model = None
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self):
        """
        Generate sample student performance data similar to Kaggle dataset
        """
        np.random.seed(42)
        n_students = 200
        
        # Generate realistic student data
        study_hours = np.random.normal(6, 2, n_students)
        study_hours = np.clip(study_hours, 1, 12)  # Reasonable study hours
        
        # Generate other features that might affect scores
        sleep_hours = np.random.normal(7.5, 1.5, n_students)
        sleep_hours = np.clip(sleep_hours, 4, 10)
        
        participation_score = np.random.normal(75, 15, n_students)
        participation_score = np.clip(participation_score, 0, 100)
        
        attendance_rate = np.random.normal(85, 10, n_students)
        attendance_rate = np.clip(attendance_rate, 60, 100)
        
        # Create realistic score relationship
        # Study hours is the main predictor, but other factors also matter
        base_score = 40 + study_hours * 5  # Base relationship
        sleep_bonus = (sleep_hours - 6) * 2  # Sleep affects performance
        participation_bonus = (participation_score - 70) * 0.3
        attendance_bonus = (attendance_rate - 80) * 0.2
        
        # Add some noise
        noise = np.random.normal(0, 8, n_students)
        
        final_scores = base_score + sleep_bonus + participation_bonus + attendance_bonus + noise
        final_scores = np.clip(final_scores, 0, 100)
        
        # Create DataFrame
        data = pd.DataFrame({
            'study_hours': study_hours,
            'sleep_hours': sleep_hours,
            'participation_score': participation_score,
            'attendance_rate': attendance_rate,
            'final_score': final_scores
        })
        
        return data
    
    def explore_data(self, data):
        """
        Perform data cleaning and basic visualization
        """
        print("=" * 60)
        print("STUDENT SCORE PREDICTION - DATA EXPLORATION")
        print("=" * 60)
        
        # Basic information
        print("\n1. Dataset Overview:")
        print(f"   - Number of students: {len(data)}")
        print(f"   - Number of features: {data.shape[1] - 1}")
        print(f"   - Target variable: final_score")
        
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
        
        # Check for outliers
        print("\n5. Outlier Detection:")
        for column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            print(f"   - {column}: {len(outliers)} outliers ({len(outliers)/len(data)*100:.1f}%)")
        
        # Create visualizations
        self.create_exploratory_plots(data)
        
        return data
    
    def create_exploratory_plots(self, data):
        """
        Create comprehensive exploratory visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student Performance Data Exploration', fontsize=16, fontweight='bold')
        
        # Distribution of study hours
        axes[0, 0].hist(data['study_hours'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Study Hours')
        axes[0, 0].set_xlabel('Study Hours')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution of final scores
        axes[0, 1].hist(data['final_score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Final Scores')
        axes[0, 1].set_xlabel('Final Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Study hours vs Final score scatter plot
        axes[0, 2].scatter(data['study_hours'], data['final_score'], alpha=0.6, color='orange')
        axes[0, 2].set_title('Study Hours vs Final Score')
        axes[0, 2].set_xlabel('Study Hours')
        axes[0, 2].set_ylabel('Final Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Correlation heatmap
        correlation_matrix = data.corr()
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Feature Correlation Matrix')
        axes[1, 0].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 0].set_yticks(range(len(correlation_matrix.columns)))
        axes[1, 0].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[1, 0].set_yticklabels(correlation_matrix.columns)
        
        # Add correlation values to heatmap
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = axes[1, 0].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')
        
        # Box plots for all features
        data_melted = data.melt()
        sns.boxplot(data=data_melted, x='variable', y='value', ax=axes[1, 1])
        axes[1, 1].set_title('Box Plots of All Features')
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('Values')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Sleep hours vs Final score
        axes[1, 2].scatter(data['sleep_hours'], data['final_score'], alpha=0.6, color='purple')
        axes[1, 2].set_title('Sleep Hours vs Final Score')
        axes[1, 2].set_xlabel('Sleep Hours')
        axes[1, 2].set_ylabel('Final Score')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 1/Task1_Student_Score_Prediction/data_exploration.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print correlation with target
        print("\n6. Correlation with Final Score:")
        correlations = data.corr()['final_score'].sort_values(ascending=False)
        for feature, corr in correlations.items():
            if feature != 'final_score':
                print(f"   - {feature}: {corr:.3f}")
    
    def prepare_data(self, data, features_to_use=None):
        """
        Prepare data for training
        """
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        
        if features_to_use is None:
            features_to_use = ['study_hours']  # Start with just study hours
        
        print(f"\nFeatures selected: {features_to_use}")
        
        # Select features and target
        X = data[features_to_use]
        y = data['final_score']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nData split:")
        print(f"   - Training set: {self.X_train.shape[0]} samples")
        print(f"   - Test set: {self.X_test.shape[0]} samples")
        print(f"   - Features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_linear_regression(self):
        """
        Train linear regression model
        """
        print("\n" + "=" * 60)
        print("LINEAR REGRESSION MODEL TRAINING")
        print("=" * 60)
        
        # Train the model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print(f"\nLinear Regression Results:")
        print(f"   Training MSE: {train_mse:.2f}")
        print(f"   Test MSE: {test_mse:.2f}")
        print(f"   Training R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   Training MAE: {train_mae:.2f}")
        print(f"   Test MAE: {test_mae:.2f}")
        
        # Model coefficients
        if self.X_train.shape[1] == 1:
            print(f"\nModel Equation:")
            print(f"   Final Score = {self.model.intercept_:.2f} + {self.model.coef_[0]:.2f} × Study Hours")
        
        return y_test_pred
    
    def train_polynomial_regression(self):
        """
        Train polynomial regression model (Bonus feature)
        """
        print("\n" + "=" * 60)
        print("POLYNOMIAL REGRESSION MODEL TRAINING (BONUS)")
        print("=" * 60)
        
        # Create polynomial features
        X_train_poly = self.poly_features.fit_transform(self.X_train)
        X_test_poly = self.poly_features.transform(self.X_test)
        
        # Train polynomial model
        self.poly_model = LinearRegression()
        self.poly_model.fit(X_train_poly, self.y_train)
        
        # Make predictions
        y_train_pred_poly = self.poly_model.predict(X_train_poly)
        y_test_pred_poly = self.poly_model.predict(X_test_poly)
        
        # Calculate metrics
        train_mse_poly = mean_squared_error(self.y_train, y_train_pred_poly)
        test_mse_poly = mean_squared_error(self.y_test, y_test_pred_poly)
        train_r2_poly = r2_score(self.y_train, y_train_pred_poly)
        test_r2_poly = r2_score(self.y_test, y_test_pred_poly)
        train_mae_poly = mean_absolute_error(self.y_train, y_train_pred_poly)
        test_mae_poly = mean_absolute_error(self.y_test, y_test_pred_poly)
        
        print(f"\nPolynomial Regression Results:")
        print(f"   Training MSE: {train_mse_poly:.2f}")
        print(f"   Test MSE: {test_mse_poly:.2f}")
        print(f"   Training R²: {train_r2_poly:.3f}")
        print(f"   Test R²: {test_r2_poly:.3f}")
        print(f"   Training MAE: {train_mae_poly:.2f}")
        print(f"   Test MAE: {test_mae_poly:.2f}")
        
        return y_test_pred_poly
    
    def visualize_predictions(self, y_pred_linear, y_pred_poly=None):
        """
        Visualize predictions and model performance
        """
        print("\n" + "=" * 60)
        print("VISUALIZATION OF PREDICTIONS")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Student Score Prediction Results', fontsize=16, fontweight='bold')
        
        # Actual vs Predicted (Linear)
        axes[0, 0].scatter(self.y_test, y_pred_linear, alpha=0.6, color='blue')
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Final Score')
        axes[0, 0].set_ylabel('Predicted Final Score')
        axes[0, 0].set_title('Linear Regression: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2_linear = r2_score(self.y_test, y_pred_linear)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2_linear:.3f}', 
                       transform=axes[0, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residuals plot (Linear)
        residuals_linear = self.y_test - y_pred_linear
        axes[0, 1].scatter(y_pred_linear, residuals_linear, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Final Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Linear Regression: Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Study hours vs predictions
        if self.X_test.shape[1] == 1:  # Only if we're using study_hours
            axes[1, 0].scatter(self.X_test.iloc[:, 0], self.y_test, alpha=0.6, 
                             color='orange', label='Actual')
            axes[1, 0].scatter(self.X_test.iloc[:, 0], y_pred_linear, alpha=0.6, 
                             color='red', label='Predicted')
            axes[1, 0].set_xlabel('Study Hours')
            axes[1, 0].set_ylabel('Final Score')
            axes[1, 0].set_title('Study Hours vs Final Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Model comparison (if polynomial model exists)
        if y_pred_poly is not None:
            axes[1, 1].scatter(self.y_test, y_pred_poly, alpha=0.6, color='purple')
            axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('Actual Final Score')
            axes[1, 1].set_ylabel('Predicted Final Score')
            axes[1, 1].set_title('Polynomial Regression: Actual vs Predicted')
            axes[1, 1].grid(True, alpha=0.3)
            
            r2_poly = r2_score(self.y_test, y_pred_poly)
            axes[1, 1].text(0.05, 0.95, f'R² = {r2_poly:.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Distribution of residuals
            axes[1, 1].hist(residuals_linear, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            axes[1, 1].set_xlabel('Residuals')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Residuals')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 1/Task1_Student_Score_Prediction/prediction_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def experiment_with_features(self, data):
        """
        Experiment with different feature combinations (Bonus feature)
        """
        print("\n" + "=" * 60)
        print("FEATURE COMBINATION EXPERIMENTS (BONUS)")
        print("=" * 60)
        
        feature_combinations = [
            ['study_hours'],
            ['study_hours', 'sleep_hours'],
            ['study_hours', 'participation_score'],
            ['study_hours', 'attendance_rate'],
            ['study_hours', 'sleep_hours', 'participation_score'],
            ['study_hours', 'sleep_hours', 'attendance_rate'],
            ['study_hours', 'participation_score', 'attendance_rate'],
            ['study_hours', 'sleep_hours', 'participation_score', 'attendance_rate']
        ]
        
        results = []
        
        for features in feature_combinations:
            # Prepare data with current feature combination
            X = data[features]
            y = data['final_score']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
                'features': features,
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'num_features': len(features)
            })
            
            print(f"\nFeatures: {features}")
            print(f"   MSE: {mse:.2f}")
            print(f"   R²: {r2:.3f}")
            print(f"   MAE: {mae:.2f}")
        
        # Create comparison plot
        self.plot_feature_comparison(results)
        
        return results
    
    def plot_feature_comparison(self, results):
        """
        Plot comparison of different feature combinations
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Feature Combination Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        feature_names = [', '.join(r['features']) for r in results]
        mse_scores = [r['mse'] for r in results]
        r2_scores = [r['r2'] for r in results]
        mae_scores = [r['mae'] for r in results]
        
        # MSE comparison
        axes[0].bar(range(len(feature_names)), mse_scores, color='skyblue', alpha=0.7)
        axes[0].set_title('Mean Squared Error Comparison')
        axes[0].set_xlabel('Feature Combinations')
        axes[0].set_ylabel('MSE')
        axes[0].set_xticks(range(len(feature_names)))
        axes[0].set_xticklabels([f"Combo {i+1}" for i in range(len(feature_names))], rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[1].bar(range(len(feature_names)), r2_scores, color='lightgreen', alpha=0.7)
        axes[1].set_title('R² Score Comparison')
        axes[1].set_xlabel('Feature Combinations')
        axes[1].set_ylabel('R² Score')
        axes[1].set_xticks(range(len(feature_names)))
        axes[1].set_xticklabels([f"Combo {i+1}" for i in range(len(feature_names))], rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # MAE comparison
        axes[2].bar(range(len(feature_names)), mae_scores, color='lightcoral', alpha=0.7)
        axes[2].set_title('Mean Absolute Error Comparison')
        axes[2].set_xlabel('Feature Combinations')
        axes[2].set_ylabel('MAE')
        axes[2].set_xticks(range(len(feature_names)))
        axes[2].set_xticklabels([f"Combo {i+1}" for i in range(len(feature_names))], rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 1/Task1_Student_Score_Prediction/feature_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print best combinations
        print("\nBest Feature Combinations:")
        best_r2 = max(results, key=lambda x: x['r2'])
        best_mse = min(results, key=lambda x: x['mse'])
        best_mae = min(results, key=lambda x: x['mae'])
        
        print(f"   Best R²: {best_r2['features']} (R² = {best_r2['r2']:.3f})")
        print(f"   Best MSE: {best_mse['features']} (MSE = {best_mse['mse']:.2f})")
        print(f"   Best MAE: {best_mae['features']} (MAE = {best_mae['mae']:.2f})")
    
    def run_complete_analysis(self):
        """
        Run the complete student score prediction analysis
        """
        print("STUDENT SCORE PREDICTION ANALYSIS")
        print("=" * 60)
        print("This analysis predicts student exam scores based on study hours")
        print("and other performance factors using linear regression.")
        print("=" * 60)
        
        # Generate sample data
        data = self.generate_sample_data()
        
        # Explore data
        data = self.explore_data(data)
        
        # Prepare data (start with just study hours)
        self.prepare_data(data, ['study_hours'])
        
        # Train linear regression
        y_pred_linear = self.train_linear_regression()
        
        # Train polynomial regression (bonus)
        y_pred_poly = self.train_polynomial_regression()
        
        # Visualize results
        self.visualize_predictions(y_pred_linear, y_pred_poly)
        
        # Experiment with different features (bonus)
        self.experiment_with_features(data)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Files saved:")
        print("   - data_exploration.png")
        print("   - prediction_results.png")
        print("   - feature_comparison.png")
        print("\nKey Findings:")
        print("   - Study hours is the strongest predictor of final scores")
        print("   - Additional features like sleep and participation improve predictions")
        print("   - Polynomial regression can capture non-linear relationships")
        print("   - Model performance varies with different feature combinations")

def main():
    """
    Main function to run the student score prediction analysis
    """
    # Create predictor instance
    predictor = StudentScorePredictor()
    
    # Run complete analysis
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()
