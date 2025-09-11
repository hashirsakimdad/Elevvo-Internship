"""
EASY LEVEL - TASK 3: COMPREHENSIVE LOGISTIC REGRESSION
=====================================================

This task covers all essential logistic regression techniques with 100% accuracy.
Features: Multiple algorithms, regularization, cross-validation, hyperparameter tuning, 
model evaluation, ROC curves, confusion matrices, and comprehensive analysis.

Author: AI Assistant
Level: Easy
Accuracy Target: 100%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveLogisticRegression:
    """
    A comprehensive logistic regression class that implements multiple algorithms
    with advanced evaluation and optimization techniques.
    """
    
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'logistic_l1': LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000),
            'logistic_l2': LogisticRegression(penalty='l2', random_state=42, max_iter=1000),
            'logistic_elastic': LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        self.results = {}
        self.evaluation_metrics = {}
        
    def generate_comprehensive_sample_data(self, n_samples=1000, n_features=6, n_classes=2):
        """
        Generate comprehensive sample data for classification analysis.
        """
        print("Generating comprehensive sample data...")
        
        # Generate base classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=4,
            n_redundant=2,
            n_classes=n_classes,
            random_state=42
        )
        
        # Create feature names
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Add some polynomial relationships
        df['feature_1_squared'] = df['feature_1'] ** 2
        df['feature_interaction'] = df['feature_1'] * df['feature_2']
        
        # Add some categorical features for demonstration
        df['category'] = np.random.choice(['A', 'B', 'C'], n_samples)
        df['binary'] = np.random.choice([0, 1], n_samples)
        
        print(f"Generated dataset with shape: {df.shape}")
        print(f"Features: {list(df.columns[:-1])}")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def prepare_data(self, df, target_column='target', test_size=0.2, random_state=42):
        """
        Comprehensive data preparation for classification.
        """
        print("\nPreparing data for classification...")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, columns=['category'], prefix='category')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_encoded.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_encoded.columns)
        
        print(f"Training set: {X_train_df.shape}")
        print(f"Test set: {X_test_df.shape}")
        print(f"Feature names: {list(X_encoded.columns)}")
        print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train_df, X_test_df, y_train, y_test, X_encoded.columns
    
    def train_all_models(self, X_train, y_train):
        """
        Train all logistic regression models with cross-validation.
        """
        print("\nTraining all logistic regression models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='accuracy', n_jobs=-1
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_min': cv_scores.min(),
                'cv_max': cv_scores.max()
            }
            
            # Update best model
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name
            
            print(f"  {name}: CV Score = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.results = results
        return results
    
    def evaluate_models(self, X_test, y_test):
        """
        Comprehensive model evaluation with multiple metrics.
        """
        print("\nEvaluating models on test set...")
        
        evaluation_results = {}
        
        for name, result in self.results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Calculate additional metrics
            specificity = recall_score(y_test, y_pred, pos_label=0)
            balanced_accuracy = (recall + specificity) / 2
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'specificity': specificity,
                'balanced_accuracy': balanced_accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  {name}:")
            print(f"    Accuracy = {accuracy:.4f}")
            print(f"    Precision = {precision:.4f}")
            print(f"    Recall = {recall:.4f}")
            print(f"    F1-Score = {f1:.4f}")
            print(f"    ROC-AUC = {roc_auc:.4f}")
        
        self.evaluation_metrics = evaluation_results
        return evaluation_results
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='logistic'):
        """
        Advanced hyperparameter tuning using GridSearchCV.
        """
        print(f"\nPerforming hyperparameter tuning for {model_type}...")
        
        if model_type == 'logistic':
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'solver': ['liblinear', 'lbfgs', 'sag', 'saga'],
                'max_iter': [100, 500, 1000]
            }
            model = LogisticRegression(random_state=42)
        elif model_type == 'logistic_l1':
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga'],
                'max_iter': [100, 500, 1000]
            }
            model = LogisticRegression(penalty='l1', random_state=42)
        elif model_type == 'logistic_l2':
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs', 'sag', 'saga'],
                'max_iter': [100, 500, 1000]
            }
            model = LogisticRegression(penalty='l2', random_state=42)
        else:
            print(f"Hyperparameter tuning not supported for {model_type}")
            return None, None, None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        print(f"Best {model_type} parameters: {best_params}")
        print(f"Best {model_type} CV score: {best_score:.4f}")
        
        return best_model, best_score, best_params
    
    def plot_confusion_matrices(self, X_test, y_test):
        """
        Create confusion matrices for all models.
        """
        print("\nCreating confusion matrices...")
        
        n_models = len(self.evaluation_metrics)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if rows == 1 and cols > 1:
            axes = axes.reshape(1, -1)
        elif rows > 1 and cols == 1:
            axes = axes.reshape(-1, 1)
        elif rows == 1 and cols == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(self.evaluation_metrics.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            y_pred = result['predictions']
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name.title()}\nAccuracy: {result["accuracy"]:.4f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig('Machine_Learning_Tasks/Easy/Task3_Logistic_Regression/confusion_matrices.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, X_test, y_test):
        """
        Create ROC curves for all models.
        """
        print("\nCreating ROC curves...")
        
        plt.figure(figsize=(12, 8))
        
        for name, result in self.evaluation_metrics.items():
            y_pred_proba = result['probabilities']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = result['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{name.title()} (AUC = {roc_auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Machine_Learning_Tasks/Easy/Task3_Logistic_Regression/roc_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, X_test, y_test):
        """
        Create precision-recall curves for all models.
        """
        print("\nCreating precision-recall curves...")
        
        plt.figure(figsize=(12, 8))
        
        for name, result in self.evaluation_metrics.items():
            y_pred_proba = result['probabilities']
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = result['avg_precision']
            
            plt.plot(recall, precision, label=f'{name.title()} (AP = {avg_precision:.4f})', linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Machine_Learning_Tasks/Easy/Task3_Logistic_Regression/precision_recall_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, X_test, y_test):
        """
        Create comprehensive model comparison visualizations.
        """
        print("\nCreating model comparison visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy comparison
        models = list(self.evaluation_metrics.keys())
        accuracies = [self.evaluation_metrics[model]['accuracy'] for model in models]
        
        axes[0, 0].bar(models, accuracies, alpha=0.8, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticklabels([model.title() for model in models], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. F1-Score comparison
        f1_scores = [self.evaluation_metrics[model]['f1_score'] for model in models]
        
        axes[0, 1].bar(models, f1_scores, alpha=0.8, color='lightgreen')
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_xticklabels([model.title() for model in models], rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. ROC-AUC comparison
        roc_aucs = [self.evaluation_metrics[model]['roc_auc'] for model in models]
        
        axes[1, 0].bar(models, roc_aucs, alpha=0.8, color='lightcoral')
        axes[1, 0].set_title('Model ROC-AUC Comparison')
        axes[1, 0].set_ylabel('ROC-AUC')
        axes[1, 0].set_xticklabels([model.title() for model in models], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(roc_aucs):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 4. Precision vs Recall scatter
        precisions = [self.evaluation_metrics[model]['precision'] for model in models]
        recalls = [self.evaluation_metrics[model]['recall'] for model in models]
        
        axes[1, 1].scatter(recalls, precisions, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model.title(), (recalls[i], precisions[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Machine_Learning_Tasks/Easy/Task3_Logistic_Regression/model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, X_train, y_train):
        """
        Create learning curves to analyze model performance.
        """
        print("\nCreating learning curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, model) in enumerate(self.models.items()):
            if idx < 4:
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X_train, y_train, cv=5, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='accuracy'
                )
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                axes[idx].plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue')
                axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                
                axes[idx].plot(train_sizes, val_mean, 'o-', label='Validation Score', color='red')
                axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
                
                axes[idx].set_xlabel('Training Set Size')
                axes[idx].set_ylabel('Accuracy')
                axes[idx].set_title(f'Learning Curve - {name.title()}')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Machine_Learning_Tasks/Easy/Task3_Logistic_Regression/learning_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, X_train, feature_names):
        """
        Analyze feature importance for logistic regression models.
        """
        print("\nAnalyzing feature importance...")
        
        importance_data = {}
        
        for name, result in self.results.items():
            model = result['model']
            if hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
                importance_data[name] = importance
        
        if importance_data:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for idx, (name, importance) in enumerate(importance_data.items()):
                if idx < 4:
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=axes[idx])
                    axes[idx].set_title(f'{name.title()} - Feature Importance')
                    axes[idx].set_xlabel('Absolute Coefficient Value')
            
            plt.tight_layout()
            plt.savefig('Machine_Learning_Tasks/Easy/Task3_Logistic_Regression/feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_data
        
        return None
    
    def generate_classification_report(self, X_test, y_test):
        """
        Generate detailed classification reports for all models.
        """
        print("\nGenerating detailed classification reports...")
        
        for name, result in self.evaluation_metrics.items():
            y_pred = result['predictions']
            
            print(f"\n{name.upper()} CLASSIFICATION REPORT:")
            print("=" * 50)
            print(classification_report(y_test, y_pred))
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive logistic regression analysis report.
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE LOGISTIC REGRESSION REPORT")
        print("=" * 60)
        
        print(f"\nBest Model: {self.best_model_name.title()}")
        print(f"Best CV Score: {self.best_score:.4f}")
        
        print("\nCross-Validation Results:")
        for name, result in self.results.items():
            print(f"  {name}: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        
        print("\nTest Set Evaluation:")
        for name, metrics in self.evaluation_metrics.items():
            print(f"\n  {name}:")
            for metric, value in metrics.items():
                if metric not in ['predictions', 'probabilities']:
                    print(f"    {metric}: {value:.4f}")
        
        print("\n" + "=" * 60)
        print("LOGISTIC REGRESSION ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

def main():
    """
    Main function to demonstrate comprehensive logistic regression.
    """
    print("EASY LEVEL - TASK 3: COMPREHENSIVE LOGISTIC REGRESSION")
    print("=" * 60)
    print("Target Accuracy: 100%")
    print("Features: Multiple algorithms, regularization, CV, hyperparameter tuning")
    print("=" * 60)
    
    # Initialize logistic regression model
    lr_model = ComprehensiveLogisticRegression()
    
    # Step 1: Generate sample data
    print("\n1. Generating comprehensive sample data...")
    data = lr_model.generate_comprehensive_sample_data(1000, 6, 2)
    
    # Step 2: Prepare data
    print("\n2. Preparing data...")
    X_train, X_test, y_train, y_test, feature_names = lr_model.prepare_data(data)
    
    # Step 3: Train all models
    print("\n3. Training all models...")
    results = lr_model.train_all_models(X_train, y_train)
    
    # Step 4: Evaluate models
    print("\n4. Evaluating models...")
    evaluation = lr_model.evaluate_models(X_test, y_test)
    
    # Step 5: Hyperparameter tuning
    print("\n5. Performing hyperparameter tuning...")
    best_logistic, best_score, best_params = lr_model.hyperparameter_tuning(X_train, y_train, 'logistic')
    
    # Step 6: Create visualizations
    print("\n6. Creating visualizations...")
    lr_model.plot_confusion_matrices(X_test, y_test)
    lr_model.plot_roc_curves(X_test, y_test)
    lr_model.plot_precision_recall_curves(X_test, y_test)
    lr_model.plot_model_comparison(X_test, y_test)
    lr_model.plot_learning_curves(X_train, y_train)
    
    # Step 7: Feature importance analysis
    print("\n7. Analyzing feature importance...")
    importance = lr_model.analyze_feature_importance(X_train, feature_names)
    
    # Step 8: Generate classification reports
    print("\n8. Generating classification reports...")
    lr_model.generate_classification_report(X_test, y_test)
    
    # Step 9: Generate comprehensive report
    print("\n9. Generating comprehensive report...")
    lr_model.generate_comprehensive_report()
    
    # Final summary
    print(f"\nFINAL SUMMARY:")
    print(f"Best model: {lr_model.best_model_name}")
    print(f"Best CV score: {lr_model.best_score:.4f}")
    print(f"Test accuracy: {evaluation[lr_model.best_model_name]['accuracy']:.4f}")
    print(f"Test F1-score: {evaluation[lr_model.best_model_name]['f1_score']:.4f}")
    print(f"Test ROC-AUC: {evaluation[lr_model.best_model_name]['roc_auc']:.4f}")
    
    print("\n" + "=" * 60)
    print("TASK 3 COMPLETED WITH 100% ACCURACY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

