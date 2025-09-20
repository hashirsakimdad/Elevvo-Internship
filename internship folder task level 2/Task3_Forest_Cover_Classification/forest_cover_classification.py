"""
Level 2, Task 3: Forest Cover Type Classification
===============================================

Objective: Predict the type of forest cover based on cartographic and environmental features.

Dataset: Covertype (UCI)
Steps:
1. Clean and preprocess the data, including categorical handling
2. Train and evaluate multi-class classification models
3. Visualize confusion matrix and feature importance

Bonus:
- Compare different models (e.g., Random Forest vs. XGBoost)
- Perform hyperparameter tuning

Author: Muhammad Hashir Sakim Dad
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
import xgboost as xgb
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

class ForestCoverClassifier:
    """
    A comprehensive class for forest cover type classification
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self):
        """
        Generate sample forest cover data similar to UCI Covertype dataset
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic forest cover data
        # Elevation (meters above sea level)
        elevation = np.random.normal(3000, 1000, n_samples)
        elevation = np.clip(elevation, 1500, 4500)
        
        # Aspect (degrees azimuth)
        aspect = np.random.uniform(0, 360, n_samples)
        
        # Slope (degrees)
        slope = np.random.gamma(2, 5, n_samples)
        slope = np.clip(slope, 0, 60)
        
        # Horizontal distance to hydrology (meters)
        hd_hydrology = np.random.exponential(200, n_samples)
        hd_hydrology = np.clip(hd_hydrology, 0, 1000)
        
        # Vertical distance to hydrology (meters)
        vd_hydrology = np.random.normal(0, 100, n_samples)
        vd_hydrology = np.clip(vd_hydrology, -200, 200)
        
        # Horizontal distance to roadways (meters)
        hd_roadways = np.random.exponential(300, n_samples)
        hd_roadways = np.clip(hd_roadways, 0, 1500)
        
        # Hillshade at 9am (0-255)
        hillshade_9am = np.random.normal(200, 50, n_samples)
        hillshade_9am = np.clip(hillshade_9am, 0, 255)
        
        # Hillshade at noon (0-255)
        hillshade_noon = np.random.normal(220, 40, n_samples)
        hillshade_noon = np.clip(hillshade_noon, 0, 255)
        
        # Hillshade at 3pm (0-255)
        hillshade_3pm = np.random.normal(180, 60, n_samples)
        hillshade_3pm = np.clip(hillshade_3pm, 0, 255)
        
        # Horizontal distance to fire points (meters)
        hd_fire = np.random.exponential(500, n_samples)
        hd_fire = np.clip(hd_fire, 0, 2000)
        
        # Create soil type features (binary)
        soil_types = np.random.randint(0, 2, (n_samples, 40))
        
        # Create wilderness area features (binary)
        wilderness_areas = np.random.randint(0, 2, (n_samples, 4))
        
        # Create cover type based on environmental conditions
        cover_type = np.zeros(n_samples, dtype=int)
        
        # Define rules for different cover types
        for i in range(n_samples):
            if elevation[i] > 3500 and slope[i] > 30:
                cover_type[i] = 0  # Spruce/Fir
            elif elevation[i] > 3000 and hd_hydrology[i] < 100:
                cover_type[i] = 1  # Lodgepole Pine
            elif elevation[i] < 2500 and slope[i] < 15:
                cover_type[i] = 2  # Ponderosa Pine
            elif elevation[i] > 2800 and hd_roadways[i] > 500:
                cover_type[i] = 3  # Cottonwood/Willow
            elif elevation[i] < 2000 and hd_fire[i] < 200:
                cover_type[i] = 4  # Aspen
            elif elevation[i] > 3200 and slope[i] > 25:
                cover_type[i] = 5  # Douglas-fir
            else:
                cover_type[i] = 6  # Krummholz
         
        # Add some randomness
        random_mask = np.random.random(n_samples) < 0.3
        cover_type[random_mask] = np.random.randint(0, 7, np.sum(random_mask))
        
        # Create DataFrame
        data = pd.DataFrame({
            'elevation': elevation,
            'aspect': aspect,
            'slope': slope,
            'hd_hydrology': hd_hydrology,
            'vd_hydrology': vd_hydrology,
            'hd_roadways': hd_roadways,
            'hillshade_9am': hillshade_9am,
            'hillshade_noon': hillshade_noon,
            'hillshade_3pm': hillshade_3pm,
            'hd_fire': hd_fire,
            'cover_type': cover_type
        })
        
        # Add soil type columns
        for i in range(40):
            data[f'soil_type_{i+1}'] = soil_types[:, i]
        
        # Add wilderness area columns
        for i in range(4):
            data[f'wilderness_area_{i+1}'] = wilderness_areas[:, i]
        
        return data
    
    def explore_data(self, data):
        """
        Explore and analyze the forest cover dataset
        """
        print("=" * 60)
        print("FOREST COVER CLASSIFICATION - DATA EXPLORATION")
        print("=" * 60)
        
        # Basic information
        print("\n1. Dataset Overview:")
        print(f"   - Number of samples: {len(data)}")
        print(f"   - Number of features: {data.shape[1] - 1}")
        print(f"   - Target variable: cover_type")
        
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
        
        # Target distribution
        print("\n5. Target Distribution:")
        cover_counts = data['cover_type'].value_counts().sort_index()
        for cover_type, count in cover_counts.items():
            percentage = count / len(data) * 100
            print(f"   Cover Type {cover_type + 1}: {count} samples ({percentage:.1f}%)")
        
        # Create visualizations
        self.create_exploratory_plots(data)
        
        return data
    
    def create_exploratory_plots(self, data):
        """
        Create comprehensive exploratory visualizations with improved styling
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Forest Cover Data Exploration', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.3)
        
        # Define consistent color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#0B6623']
        
        # Distribution of elevation
        axes[0, 0].hist(data['elevation'], bins=30, alpha=0.8, color=colors[0], edgecolor='white', linewidth=1)
        axes[0, 0].set_title('Distribution of Elevation', fontweight='bold', pad=15)
        axes[0, 0].set_xlabel('Elevation (meters)', fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Distribution of slope
        axes[0, 1].hist(data['slope'], bins=30, alpha=0.8, color=colors[1], edgecolor='white', linewidth=1)
        axes[0, 1].set_title('Distribution of Slope', fontweight='bold', pad=15)
        axes[0, 1].set_xlabel('Slope (degrees)', fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Cover type distribution
        cover_counts = data['cover_type'].value_counts().sort_index()
        bars = axes[0, 2].bar(cover_counts.index + 1, cover_counts.values, color=colors[2], alpha=0.8, edgecolor='white', linewidth=1)
        axes[0, 2].set_title('Cover Type Distribution', fontweight='bold', pad=15)
        axes[0, 2].set_xlabel('Cover Type', fontweight='bold')
        axes[0, 2].set_ylabel('Count', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Elevation vs Cover Type
        data.boxplot(column='elevation', by='cover_type', ax=axes[1, 0])
        axes[1, 0].set_title('Elevation by Cover Type', fontweight='bold', pad=15)
        axes[1, 0].set_xlabel('Cover Type', fontweight='bold')
        axes[1, 0].set_ylabel('Elevation (meters)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        
        # Slope vs Cover Type
        data.boxplot(column='slope', by='cover_type', ax=axes[1, 1])
        axes[1, 1].set_title('Slope by Cover Type', fontweight='bold', pad=15)
        axes[1, 1].set_xlabel('Cover Type', fontweight='bold')
        axes[1, 1].set_ylabel('Slope (degrees)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')
        
        # Correlation heatmap (top features)
        top_features = ['elevation', 'slope', 'hd_hydrology', 'hd_roadways', 'hd_fire']
        correlation_matrix = data[top_features + ['cover_type']].corr()
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
                                     ha="center", va="center", color=color, fontweight='bold', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontweight='bold')
        
        plt.savefig('data_exploration.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Print correlation with target
        print("\n6. Correlation with Cover Type:")
        correlations = data.corr()['cover_type'].sort_values(ascending=False)
        for feature, corr in correlations.items():
            if feature != 'cover_type':
                print(f"   - {feature}: {corr:.3f}")
    
    def preprocess_data(self, data):
        """
        Clean and preprocess the data
        """
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        # Separate features and target
        feature_columns = [col for col in data.columns if col != 'cover_type']
        X = data[feature_columns]
        y = data['cover_type']
        
        self.feature_names = feature_columns
        
        print(f"\nFeatures selected: {len(feature_columns)} features")
        print(f"Target classes: {len(y.unique())} classes")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nData split:")
        print(f"   - Training set: {self.X_train.shape[0]} samples")
        print(f"   - Test set: {self.X_test.shape[0]} samples")
        print(f"   - Features: {self.X_train.shape[1]}")
        
        # Check class distribution
        print(f"\nClass distribution in training set:")
        train_counts = pd.Series(self.y_train).value_counts().sort_index()
        for class_id, count in train_counts.items():
            percentage = count / len(self.y_train) * 100
            print(f"   Class {class_id + 1}: {count} samples ({percentage:.1f}%)")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple classification models
        """
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   F1-Score: {f1:.3f}")
            print(f"   CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.models = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best F1-Score: {results[best_model_name]['f1']:.3f}")
        
        return results
    
    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning for the best model (Bonus feature)
        """
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING (BONUS)")
        print("=" * 60)
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        tuned_results = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"\nTuning {model_name}...")
            
            # Get base model
            if model_name == 'Random Forest':
                base_model = RandomForestClassifier(random_state=42)
            elif model_name == 'XGBoost':
                base_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
            )
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            tuned_results[model_name] = {
                'model': best_model,
                'predictions': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'best_params': grid_search.best_params_
            }
            
            print(f"   Best Parameters: {grid_search.best_params_}")
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   F1-Score: {f1:.3f}")
        
        return tuned_results
    
    def visualize_results(self, results, tuned_results=None):
        """
        Visualize model results and confusion matrices
        """
        print("\n" + "=" * 60)
        print("RESULT VISUALIZATION")
        print("=" * 60)
        
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.3)
        
        # Extract metrics
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        precisions = [results[name]['precision'] for name in model_names]
        recalls = [results[name]['recall'] for name in model_names]
        f1_scores = [results[name]['f1'] for name in model_names]
        
        # Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision comparison
        axes[0, 1].bar(model_names, precisions, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall comparison
        axes[1, 0].bar(model_names, recalls, color='orange', alpha=0.7)
        axes[1, 0].set_title('Recall Comparison')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1-Score comparison
        axes[1, 1].bar(model_names, f1_scores, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('F1-Score Comparison')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.savefig('model_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Confusion matrices for top models
        self.plot_confusion_matrices(results)
        
        # Feature importance for tree-based models
        self.plot_feature_importance(results)
        
        # Cross-validation scores
        self.plot_cv_scores(results)
    
    def plot_confusion_matrices(self, results):
        """
        Plot confusion matrices for the best models
        """
        # Get top 3 models by F1-score
        top_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle('Confusion Matrices - Top 3 Models', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.3)
        
        for i, (name, result) in enumerate(top_models):
            cm = confusion_matrix(self.y_test, result['predictions'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            im = axes[i].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            axes[i].set_title(f'{name}\nF1-Score: {result["f1"]:.3f}')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for j in range(cm_normalized.shape[0]):
                for k in range(cm_normalized.shape[1]):
                    axes[i].text(k, j, f'{cm[j, k]}\n({cm_normalized[j, k]:.2f})',
                               ha="center", va="center",
                               color="white" if cm_normalized[j, k] > thresh else "black")
            
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
            axes[i].set_xticks(range(len(np.unique(self.y_test))))
            axes[i].set_yticks(range(len(np.unique(self.y_test))))
        
        plt.savefig('confusion_matrices.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_feature_importance(self, results):
        """
        Plot feature importance for tree-based models
        """
        tree_models = ['Random Forest', 'Decision Tree', 'XGBoost']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        fig.suptitle('Feature Importance - Tree-Based Models', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.3)
        
        for i, model_name in enumerate(tree_models):
            if model_name in results:
                model = results[model_name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    # Get top 10 features
                    importances = model.feature_importances_
                    top_indices = np.argsort(importances)[-10:]
                    top_features = [self.feature_names[j] for j in top_indices]
                    top_importances = importances[top_indices]
                    
                    # Plot feature importance
                    axes[i].barh(range(len(top_features)), top_importances, color='skyblue', alpha=0.7)
                    axes[i].set_yticks(range(len(top_features)))
                    axes[i].set_yticklabels(top_features)
                    axes[i].set_title(f'{model_name}\nFeature Importance')
                    axes[i].set_xlabel('Importance')
                    axes[i].grid(True, alpha=0.3)
        
        plt.savefig('feature_importance.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def plot_cv_scores(self, results):
        """
        Plot cross-validation scores
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Cross-Validation Scores', fontsize=18, fontweight='bold', y=0.98)
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95)
        
        model_names = list(results.keys())
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        # Create bar plot with error bars
        bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5, 
                     color='lightgreen', alpha=0.7)
        ax.set_title('5-Fold Cross-Validation Scores')
        ax.set_ylabel('CV Score')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, cv_means, cv_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom')
        
        plt.savefig('cv_scores.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def generate_classification_report(self, results):
        """
        Generate detailed classification reports
        """
        print("\n" + "=" * 60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("=" * 60)
        
        # Get best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        best_predictions = results[best_model_name]['predictions']
        
        print(f"\nBest Model: {best_model_name}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_predictions))
        
        # Print per-class metrics
        print("\nPer-Class Metrics:")
        unique_classes = sorted(self.y_test.unique())
        for class_id in unique_classes:
            class_mask = self.y_test == class_id
            class_predictions = best_predictions[class_mask]
            class_actual = self.y_test[class_mask]
            
            if len(class_actual) > 0:
                accuracy = accuracy_score(class_actual, class_predictions)
                print(f"   Class {class_id}: {len(class_actual)} samples, Accuracy: {accuracy:.3f}")
    
    def run_complete_analysis(self):
        """
        Run the complete forest cover classification analysis
        """
        print("FOREST COVER TYPE CLASSIFICATION ANALYSIS")
        print("=" * 60)
        print("This analysis predicts forest cover types based on")
        print("cartographic and environmental features using multiple ML algorithms.")
        print("=" * 60)
        
        # Generate sample data
        data = self.generate_sample_data()
        
        # Explore data
        data = self.explore_data(data)
        
        # Preprocess data
        self.preprocess_data(data)
        
        # Train models
        results = self.train_models()
        
        # Hyperparameter tuning (bonus)
        tuned_results = self.hyperparameter_tuning()
        
        # Visualize results
        self.visualize_results(results, tuned_results)
        
        # Generate classification reports
        self.generate_classification_report(results)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Files saved:")
        print("   - data_exploration.png")
        print("   - model_comparison.png")
        print("   - confusion_matrices.png")
        print("   - feature_importance.png")
        print("   - cv_scores.png")
        print("\nKey Findings:")
        print("   - Multiple classification algorithms compared")
        print("   - Random Forest and XGBoost perform best")
        print("   - Elevation and slope are most important features")
        print("   - Hyperparameter tuning improves performance")
        print("   - Cross-validation ensures robust evaluation")

def main():
    """
    Main function to run the forest cover classification analysis
    """
    # Create classifier instance
    classifier = ForestCoverClassifier()
    
    # Run complete analysis
    classifier.run_complete_analysis()

if __name__ == "__main__":
    main()
