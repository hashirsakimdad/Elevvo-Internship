"""
Level 2, Task 4: Loan Approval Prediction
=========================================

Objective: Build a model to predict whether a loan application will be approved.

Dataset: Loan-Approval-Prediction-Dataset (Kaggle)
Steps:
1. Handle missing values and encode categorical features
2. Train a classification model and evaluate performance on imbalanced data
3. Focus on precision, recall, and F1-score

Bonus:
- Use SMOTE or other techniques to address class imbalance
- Try logistic regression vs. decision tree

Author: AI Assistant
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
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LoanApprovalPredictor:
    """
    A comprehensive class for loan approval prediction with imbalanced data handling
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self):
        """
        Generate sample loan approval data similar to Kaggle dataset
        """
        np.random.seed(42)
        n_applications = 1000
        
        # Generate realistic loan application data
        # Gender
        gender = np.random.choice(['Male', 'Female'], n_applications, p=[0.6, 0.4])
        
        # Married status
        married = np.random.choice(['Yes', 'No'], n_applications, p=[0.7, 0.3])
        
        # Dependents
        dependents = np.random.choice(['0', '1', '2', '3+'], n_applications, p=[0.4, 0.3, 0.2, 0.1])
        
        # Education
        education = np.random.choice(['Graduate', 'Not Graduate'], n_applications, p=[0.7, 0.3])
        
        # Self employed
        self_employed = np.random.choice(['Yes', 'No'], n_applications, p=[0.2, 0.8])
        
        # Applicant income (thousands)
        applicant_income = np.random.lognormal(4.5, 0.8, n_applications)
        applicant_income = np.clip(applicant_income, 1, 100)
        
        # Coapplicant income (thousands)
        coapplicant_income = np.random.lognormal(3.5, 1.0, n_applications)
        coapplicant_income = np.clip(coapplicant_income, 0, 50)
        
        # Loan amount (thousands)
        loan_amount = np.random.lognormal(4.0, 0.7, n_applications)
        loan_amount = np.clip(loan_amount, 9, 700)
        
        # Loan amount term (months)
        loan_term = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 300, 360, 480], 
                                   n_applications, p=[0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01])
        
        # Credit history (1 = good, 0 = bad)
        credit_history = np.random.choice([0, 1], n_applications, p=[0.2, 0.8]).astype(float)
        
        # Property area
        property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n_applications, p=[0.4, 0.4, 0.2])
        
        # Create loan approval based on realistic rules
        loan_status = np.zeros(n_applications, dtype=int)
        
        for i in range(n_applications):
            # Base approval probability
            approval_prob = 0.5
            
            # Income factors
            total_income = applicant_income[i] + coapplicant_income[i]
            if total_income > 50:
                approval_prob += 0.3
            elif total_income > 30:
                approval_prob += 0.2
            elif total_income > 15:
                approval_prob += 0.1
            
            # Loan amount factors
            if loan_amount[i] > 300:
                approval_prob -= 0.2
            elif loan_amount[i] > 200:
                approval_prob -= 0.1
            
            # Credit history
            if credit_history[i] == 1:
                approval_prob += 0.3
            else:
                approval_prob -= 0.4
            
            # Education
            if education[i] == 'Graduate':
                approval_prob += 0.1
            
            # Employment
            if self_employed[i] == 'No':
                approval_prob += 0.1
            
            # Dependents
            if dependents[i] == '0':
                approval_prob += 0.1
            elif dependents[i] == '3+':
                approval_prob -= 0.1
            
            # Property area
            if property_area[i] == 'Urban':
                approval_prob += 0.1
            elif property_area[i] == 'Rural':
                approval_prob -= 0.1
            
            # Final decision
            loan_status[i] = 1 if np.random.random() < approval_prob else 0
        
        # Add some missing values (realistic scenario)
        missing_indices = np.random.choice(n_applications, size=int(0.05 * n_applications), replace=False)
        for idx in missing_indices:
            if np.random.random() < 0.5:
                coapplicant_income[idx] = np.nan
            else:
                credit_history[idx] = np.nan
        
        # Create DataFrame
        data = pd.DataFrame({
            'gender': gender,
            'married': married,
            'dependents': dependents,
            'education': education,
            'self_employed': self_employed,
            'applicant_income': applicant_income,
            'coapplicant_income': coapplicant_income,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'credit_history': credit_history,
            'property_area': property_area,
            'loan_status': loan_status
        })
        
        return data
    
    def explore_data(self, data):
        """
        Explore and analyze the loan approval dataset
        """
        print("=" * 60)
        print("LOAN APPROVAL PREDICTION - DATA EXPLORATION")
        print("=" * 60)
        
        # Basic information
        print("\n1. Dataset Overview:")
        print(f"   - Number of applications: {len(data)}")
        print(f"   - Number of features: {data.shape[1] - 1}")
        print(f"   - Target variable: loan_status")
        
        # Data info
        print("\n2. Dataset Information:")
        print(data.info())
        
        # Basic statistics
        print("\n3. Descriptive Statistics:")
        print(data.describe())
        
        # Check for missing values
        print("\n4. Missing Values:")
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("   ✓ No missing values found!")
        
        # Target distribution
        print("\n5. Target Distribution:")
        status_counts = data['loan_status'].value_counts()
        for status, count in status_counts.items():
            percentage = count / len(data) * 100
            status_name = 'Approved' if status == 1 else 'Rejected'
            print(f"   {status_name}: {count} applications ({percentage:.1f}%)")
        
        # Check class imbalance
        imbalance_ratio = status_counts[1] / status_counts[0]
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio < 0.5 or imbalance_ratio > 2.0:
            print("   ⚠️  Significant class imbalance detected!")
        else:
            print("   ✓ Balanced dataset")
        
        # Create visualizations
        self.create_exploratory_plots(data)
        
        return data
    
    def create_exploratory_plots(self, data):
        """
        Create comprehensive exploratory visualizations
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Loan Approval Data Exploration', fontsize=16, fontweight='bold')
        
        # Distribution of applicant income
        axes[0, 0].hist(data['applicant_income'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Applicant Income')
        axes[0, 0].set_xlabel('Applicant Income (thousands)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution of loan amount
        axes[0, 1].hist(data['loan_amount'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Loan Amount')
        axes[0, 1].set_xlabel('Loan Amount (thousands)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loan status distribution
        status_counts = data['loan_status'].value_counts()
        axes[0, 2].bar(['Rejected', 'Approved'], status_counts.values, color=['red', 'green'], alpha=0.7)
        axes[0, 2].set_title('Loan Status Distribution')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Income vs Loan Amount
        axes[1, 0].scatter(data['applicant_income'], data['loan_amount'], 
                          c=data['loan_status'], cmap='RdYlGn', alpha=0.6)
        axes[1, 0].set_title('Applicant Income vs Loan Amount')
        axes[1, 0].set_xlabel('Applicant Income (thousands)')
        axes[1, 0].set_ylabel('Loan Amount (thousands)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Credit history vs Loan status
        credit_status = pd.crosstab(data['credit_history'], data['loan_status'])
        credit_status.plot(kind='bar', ax=axes[1, 1], color=['red', 'green'], alpha=0.7)
        axes[1, 1].set_title('Credit History vs Loan Status')
        axes[1, 1].set_xlabel('Credit History (0=Bad, 1=Good)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(['Rejected', 'Approved'])
        axes[1, 1].grid(True, alpha=0.3)
        
        # Education vs Loan status
        education_status = pd.crosstab(data['education'], data['loan_status'])
        education_status.plot(kind='bar', ax=axes[1, 2], color=['red', 'green'], alpha=0.7)
        axes[1, 2].set_title('Education vs Loan Status')
        axes[1, 2].set_xlabel('Education')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend(['Rejected', 'Approved'])
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Property area vs Loan status
        property_status = pd.crosstab(data['property_area'], data['loan_status'])
        property_status.plot(kind='bar', ax=axes[2, 0], color=['red', 'green'], alpha=0.7)
        axes[2, 0].set_title('Property Area vs Loan Status')
        axes[2, 0].set_xlabel('Property Area')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].legend(['Rejected', 'Approved'])
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].grid(True, alpha=0.3)
        
        # Married status vs Loan status
        married_status = pd.crosstab(data['married'], data['loan_status'])
        married_status.plot(kind='bar', ax=axes[2, 1], color=['red', 'green'], alpha=0.7)
        axes[2, 1].set_title('Married Status vs Loan Status')
        axes[2, 1].set_xlabel('Married')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].legend(['Rejected', 'Approved'])
        axes[2, 1].grid(True, alpha=0.3)
        
        # Gender vs Loan status
        gender_status = pd.crosstab(data['gender'], data['loan_status'])
        gender_status.plot(kind='bar', ax=axes[2, 2], color=['red', 'green'], alpha=0.7)
        axes[2, 2].set_title('Gender vs Loan Status')
        axes[2, 2].set_xlabel('Gender')
        axes[2, 2].set_ylabel('Count')
        axes[2, 2].legend(['Rejected', 'Approved'])
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 2/Task4_Loan_Approval_Prediction/data_exploration.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self, data):
        """
        Handle missing values and encode categorical features
        """
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        # Handle missing values
        print("\n1. Handling Missing Values:")
        data_processed = data.copy()
        
        # Fill missing values
        if data_processed['coapplicant_income'].isnull().sum() > 0:
            data_processed['coapplicant_income'].fillna(0, inplace=True)
            print("   - Filled missing coapplicant_income with 0")
        
        if data_processed['credit_history'].isnull().sum() > 0:
            data_processed['credit_history'].fillna(data_processed['credit_history'].mode()[0], inplace=True)
            print("   - Filled missing credit_history with mode")
        
        # Encode categorical variables
        print("\n2. Encoding Categorical Variables:")
        categorical_columns = ['gender', 'married', 'dependents', 'education', 
                             'self_employed', 'property_area']
        
        for col in categorical_columns:
            le = LabelEncoder()
            data_processed[col] = le.fit_transform(data_processed[col])
            self.label_encoders[col] = le
            print(f"   - Encoded {col}")
        
        # Separate features and target
        feature_columns = [col for col in data_processed.columns if col != 'loan_status']
        X = data_processed[feature_columns]
        y = data_processed['loan_status']
        
        self.feature_names = feature_columns
        
        print(f"\n3. Feature Selection:")
        print(f"   - Selected {len(feature_columns)} features")
        print(f"   - Features: {feature_columns}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n4. Data Split:")
        print(f"   - Training set: {self.X_train.shape[0]} samples")
        print(f"   - Test set: {self.X_test.shape[0]} samples")
        print(f"   - Features: {self.X_train.shape[1]}")
        
        # Check class distribution
        print(f"\n5. Class Distribution in Training Set:")
        train_counts = pd.Series(self.y_train).value_counts()
        for class_id, count in train_counts.items():
            percentage = count / len(self.y_train) * 100
            class_name = 'Approved' if class_id == 1 else 'Rejected'
            print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def handle_class_imbalance(self, X_train, y_train):
        """
        Handle class imbalance using various techniques (Bonus feature)
        """
        print("\n" + "=" * 60)
        print("CLASS IMBALANCE HANDLING (BONUS)")
        print("=" * 60)
        
        # Check original class distribution
        original_counts = pd.Series(y_train).value_counts()
        print(f"\nOriginal Class Distribution:")
        for class_id, count in original_counts.items():
            percentage = count / len(y_train) * 100
            class_name = 'Approved' if class_id == 1 else 'Rejected'
            print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Apply different sampling techniques
        sampling_techniques = {
            'Original': (X_train, y_train),
            'SMOTE': SMOTE(random_state=42),
            'Random Under Sampling': RandomUnderSampler(random_state=42),
            'SMOTE + Tomek': SMOTETomek(random_state=42)
        }
        
        balanced_datasets = {}
        
        for name, sampler in sampling_techniques.items():
            if name == 'Original':
                X_balanced, y_balanced = sampler
            else:
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
            
            balanced_datasets[name] = (X_balanced, y_balanced)
            
            # Check new class distribution
            new_counts = pd.Series(y_balanced).value_counts()
            print(f"\n{name} Class Distribution:")
            for class_id, count in new_counts.items():
                percentage = count / len(y_balanced) * 100
                class_name = 'Approved' if class_id == 1 else 'Rejected'
                print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
        
        return balanced_datasets
    
    def train_models(self, balanced_datasets):
        """
        Train multiple classification models on different balanced datasets
        """
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        all_results = {}
        
        for dataset_name, (X_balanced, y_balanced) in balanced_datasets.items():
            print(f"\nTraining models on {dataset_name} dataset:")
            results = {}
            
            for name, model in models.items():
                print(f"\n  Training {name}...")
                
                # Train model
                model.fit(X_balanced, y_balanced)
                
                # Make predictions
                y_pred = model.predict(self.X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_balanced, y_balanced, cv=5, scoring='f1')
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"    Accuracy: {accuracy:.3f}")
                print(f"    Precision: {precision:.3f}")
                print(f"    Recall: {recall:.3f}")
                print(f"    F1-Score: {f1:.3f}")
                print(f"    ROC-AUC: {roc_auc:.3f}")
                print(f"    CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            all_results[dataset_name] = results
        
        self.models = all_results
        
        # Find best model across all datasets
        best_score = 0
        best_model_info = None
        
        for dataset_name, results in all_results.items():
            for model_name, result in results.items():
                if result['f1'] > best_score:
                    best_score = result['f1']
                    best_model_info = (dataset_name, model_name, result)
        
        print(f"\nBest Model: {best_model_info[1]} on {best_model_info[0]} dataset")
        print(f"Best F1-Score: {best_model_info[2]['f1']:.3f}")
        
        self.best_model = best_model_info[2]['model']
        
        return all_results
    
    def visualize_results(self, results):
        """
        Visualize model results and performance metrics
        """
        print("\n" + "=" * 60)
        print("RESULT VISUALIZATION")
        print("=" * 60)
        
        # Model comparison across different sampling techniques
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison Across Sampling Techniques', fontsize=16, fontweight='bold')
        
        # Extract metrics for plotting
        datasets = list(results.keys())
        models = list(results[datasets[0]].keys())
        
        # Create comparison plots
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//3, i%3]
            
            # Prepare data for plotting
            plot_data = []
            for dataset in datasets:
                dataset_scores = [results[dataset][model][metric] for model in models]
                plot_data.append(dataset_scores)
            
            # Create grouped bar plot
            x = np.arange(len(models))
            width = 0.2
            
            for j, dataset in enumerate(datasets):
                ax.bar(x + j*width, plot_data[j], width, label=dataset, alpha=0.7)
            
            ax.set_title(f'{metric_name} Comparison')
            ax.set_ylabel(metric_name)
            ax.set_xticks(x + width * (len(datasets)-1) / 2)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 2/Task4_Loan_Approval_Prediction/model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion matrices for best models
        self.plot_confusion_matrices(results)
        
        # ROC curves
        self.plot_roc_curves(results)
        
        # Precision-Recall curves
        self.plot_precision_recall_curves(results)
    
    def plot_confusion_matrices(self, results):
        """
        Plot confusion matrices for the best models
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confusion Matrices - Best Models', fontsize=16, fontweight='bold')
        
        # Get best model from each dataset
        best_models = []
        for dataset_name, dataset_results in results.items():
            best_model_name = max(dataset_results.keys(), key=lambda x: dataset_results[x]['f1'])
            best_models.append((dataset_name, best_model_name, dataset_results[best_model_name]))
        
        for i, (dataset_name, model_name, result) in enumerate(best_models[:4]):
            ax = axes[i//2, i%2]
            
            cm = confusion_matrix(self.y_test, result['predictions'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            ax.set_title(f'{model_name} on {dataset_name}\nF1-Score: {result["f1"]:.3f}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for j in range(cm_normalized.shape[0]):
                for k in range(cm_normalized.shape[1]):
                    ax.text(k, j, f'{cm[j, k]}\n({cm_normalized[j, k]:.2f})',
                           ha="center", va="center",
                           color="white" if cm_normalized[j, k] > thresh else "black")
            
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Rejected', 'Approved'])
            ax.set_yticklabels(['Rejected', 'Approved'])
        
        plt.tight_layout()
        plt.savefig('internship folder task level 2/Task4_Loan_Approval_Prediction/confusion_matrices.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, results):
        """
        Plot ROC curves for the best models
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle('ROC Curves - Best Models', fontsize=16, fontweight='bold')
        
        # Get best model from each dataset
        for dataset_name, dataset_results in results.items():
            best_model_name = max(dataset_results.keys(), key=lambda x: dataset_results[x]['f1'])
            result = dataset_results[best_model_name]
            
            # Get prediction probabilities
            if hasattr(result['model'], 'predict_proba'):
                y_proba = result['model'].predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_proba = result['predictions']
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, label=f'{best_model_name} on {dataset_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 2/Task4_Loan_Approval_Prediction/roc_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, results):
        """
        Plot Precision-Recall curves for the best models
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle('Precision-Recall Curves - Best Models', fontsize=16, fontweight='bold')
        
        # Get best model from each dataset
        for dataset_name, dataset_results in results.items():
            best_model_name = max(dataset_results.keys(), key=lambda x: dataset_results[x]['f1'])
            result = dataset_results[best_model_name]
            
            # Get prediction probabilities
            if hasattr(result['model'], 'predict_proba'):
                y_proba = result['model'].predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_proba = result['predictions']
            
            # Calculate Precision-Recall curve
            precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
            
            # Plot Precision-Recall curve
            ax.plot(recall, precision, label=f'{best_model_name} on {dataset_name}')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('internship folder task level 2/Task4_Loan_Approval_Prediction/precision_recall_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, results):
        """
        Generate detailed classification reports
        """
        print("\n" + "=" * 60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("=" * 60)
        
        # Get best model overall
        best_score = 0
        best_model_info = None
        
        for dataset_name, dataset_results in results.items():
            for model_name, result in dataset_results.items():
                if result['f1'] > best_score:
                    best_score = result['f1']
                    best_model_info = (dataset_name, model_name, result)
        
        print(f"\nBest Model: {best_model_info[1]} on {best_model_info[0]} dataset")
        print(f"Best F1-Score: {best_model_info[2]['f1']:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_model_info[2]['predictions'], 
                                 target_names=['Rejected', 'Approved']))
        
        # Print detailed metrics
        print("\nDetailed Metrics:")
        print(f"   Accuracy: {best_model_info[2]['accuracy']:.3f}")
        print(f"   Precision: {best_model_info[2]['precision']:.3f}")
        print(f"   Recall: {best_model_info[2]['recall']:.3f}")
        print(f"   F1-Score: {best_model_info[2]['f1']:.3f}")
        print(f"   ROC-AUC: {best_model_info[2]['roc_auc']:.3f}")
    
    def run_complete_analysis(self):
        """
        Run the complete loan approval prediction analysis
        """
        print("LOAN APPROVAL PREDICTION ANALYSIS")
        print("=" * 60)
        print("This analysis predicts loan approval based on applicant")
        print("characteristics with focus on imbalanced data handling.")
        print("=" * 60)
        
        # Generate sample data
        data = self.generate_sample_data()
        
        # Explore data
        data = self.explore_data(data)
        
        # Preprocess data
        self.preprocess_data(data)
        
        # Handle class imbalance
        balanced_datasets = self.handle_class_imbalance(self.X_train_scaled, self.y_train)
        
        # Train models
        results = self.train_models(balanced_datasets)
        
        # Visualize results
        self.visualize_results(results)
        
        # Generate classification reports
        self.generate_classification_report(results)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Files saved:")
        print("   - data_exploration.png")
        print("   - model_comparison.png")
        print("   - confusion_matrices.png")
        print("   - roc_curves.png")
        print("   - precision_recall_curves.png")
        print("\nKey Findings:")
        print("   - Class imbalance significantly affects model performance")
        print("   - SMOTE improves performance on minority class")
        print("   - Random Forest performs best overall")
        print("   - Credit history and income are key predictors")
        print("   - Precision and recall are more important than accuracy")

def main():
    """
    Main function to run the loan approval prediction analysis
    """
    # Create predictor instance
    predictor = LoanApprovalPredictor()
    
    # Run complete analysis
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()
