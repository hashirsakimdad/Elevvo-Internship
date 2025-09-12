# Level 2 Machine Learning Tasks
=================================

This directory contains Level 2 machine learning tasks designed for intermediate learners. Each task includes comprehensive implementations with advanced techniques, detailed documentation, and professional visualizations.

## 📁 Task Structure

```
Level 2/
├── Task3_Forest_Cover_Classification/
│   └── forest_cover_classification.py
├── Task4_Loan_Approval_Prediction/
│   └── loan_approval_prediction.py
├── Task5_Movie_Recommendation_System/
│   └── movie_recommendation_system.py
└── README.md
```

## 🎯 Learning Objectives

By completing these tasks, you will master:

- **Multi-class Classification**: Forest cover type prediction with advanced algorithms
- **Imbalanced Data Handling**: Loan approval prediction with SMOTE and sampling techniques
- **Recommendation Systems**: Collaborative filtering and matrix factorization

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### Running Individual Tasks

```bash
# Task 3: Forest Cover Classification
python Task3_Forest_Cover_Classification/forest_cover_classification.py

# Task 4: Loan Approval Prediction
python Task4_Loan_Approval_Prediction/loan_approval_prediction.py

# Task 5: Movie Recommendation System
python Task5_Movie_Recommendation_System/movie_recommendation_system.py
```

## 📊 Task Descriptions

### Task 3: Forest Cover Type Classification
**File**: `Task3_Forest_Cover_Classification/forest_cover_classification.py`

**Objective**: Predict the type of forest cover based on cartographic and environmental features.

**Features**:
- Multi-class classification (7 forest cover types)
- Multiple algorithms (Random Forest, XGBoost, SVM, Logistic Regression)
- Hyperparameter tuning with GridSearchCV
- Comprehensive model evaluation and comparison
- Feature importance analysis
- Confusion matrices and classification reports

**Key Methods**:
- `generate_sample_data()`: Creates realistic forest cover data
- `explore_data()`: Comprehensive data analysis and visualization
- `preprocess_data()`: Data preprocessing and feature scaling
- `train_models()`: Multiple model training and evaluation
- `hyperparameter_tuning()`: Advanced hyperparameter optimization
- `visualize_results()`: Model comparison and performance visualization
- `plot_confusion_matrices()`: Confusion matrix visualization
- `plot_feature_importance()`: Feature importance analysis

**Output**: Data exploration plots, model comparison, confusion matrices, feature importance

### Task 4: Loan Approval Prediction
**File**: `Task4_Loan_Approval_Prediction/loan_approval_prediction.py`

**Objective**: Build a model to predict whether a loan application will be approved.

**Features**:
- Binary classification with imbalanced data handling
- Multiple sampling techniques (SMOTE, Random Under Sampling, SMOTE+Tomek)
- Focus on precision, recall, and F1-score metrics
- ROC curves and Precision-Recall curves
- Comprehensive model comparison across sampling techniques
- Detailed classification reports

**Key Methods**:
- `generate_sample_data()`: Creates realistic loan application data
- `explore_data()`: Data exploration with class imbalance analysis
- `preprocess_data()`: Missing value handling and categorical encoding
- `handle_class_imbalance()`: Advanced sampling techniques
- `train_models()`: Model training on different balanced datasets
- `visualize_results()`: Performance comparison across sampling techniques
- `plot_roc_curves()`: ROC curve analysis
- `plot_precision_recall_curves()`: Precision-recall analysis

**Output**: Data exploration plots, model comparison, confusion matrices, ROC curves

### Task 5: Movie Recommendation System
**File**: `Task5_Movie_Recommendation_System/movie_recommendation_system.py`

**Objective**: Build a system that recommends movies based on user similarity.

**Features**:
- User-based collaborative filtering
- Item-based collaborative filtering (bonus)
- Matrix factorization using SVD (bonus)
- Precision at K evaluation
- Recommendation method comparison
- User-item matrix analysis
- Genre-based analysis

**Key Methods**:
- `generate_sample_data()`: Creates realistic movie rating data
- `explore_data()`: Data exploration and sparsity analysis
- `create_user_item_matrix()`: User-item matrix creation
- `user_based_collaborative_filtering()`: User-based recommendations
- `item_based_collaborative_filtering()`: Item-based recommendations
- `matrix_factorization_svd()`: SVD-based recommendations
- `evaluate_recommendations()`: Precision at K evaluation
- `compare_recommendation_methods()`: Method comparison

**Output**: Data exploration plots, recommendation comparisons, evaluation metrics

## 🔧 Key Features

### Advanced Implementation
- **Complete Code**: Full working implementations with detailed comments
- **Sample Data**: Built-in data generation for demonstration
- **Professional Output**: High-quality plots and analysis
- **Error Handling**: Robust error handling and validation

### Advanced Techniques
- **Hyperparameter Tuning**: GridSearchCV for optimal parameter selection
- **Class Imbalance Handling**: SMOTE, Random Under Sampling, SMOTE+Tomek
- **Collaborative Filtering**: User-based, item-based, and matrix factorization
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Cross-Validation**: Robust model evaluation

### Professional Visualizations
- **Multiple Plot Types**: Confusion matrices, ROC curves, feature importance
- **Custom Styling**: Professional appearance with custom colors and fonts
- **Export Options**: High-resolution PNG images
- **Comprehensive Analysis**: Multi-panel visualizations and dashboards

## 📈 Output Examples

Each task generates:

### Console Output
- Detailed progress and results
- Performance metrics and statistics
- Model comparison results
- Comprehensive analysis reports

### Visualizations
- Professional plots and charts
- High-resolution images (300 DPI)
- Comprehensive analysis dashboards
- Algorithm comparison plots

### Saved Files
- PNG images (300 DPI)
- Comprehensive analysis reports
- Model performance metrics

## 🎯 Learning Path

### Recommended Order
1. **Task 3**: Forest Cover Classification (Multi-class Classification)
2. **Task 4**: Loan Approval Prediction (Imbalanced Data)
3. **Task 5**: Movie Recommendation System (Recommendation Systems)

### Prerequisites
- Intermediate Python knowledge
- Understanding of machine learning concepts
- Familiarity with scikit-learn
- Basic understanding of statistics

### Skills Gained
- Multi-class classification
- Imbalanced data handling
- Collaborative filtering
- Matrix factorization
- Hyperparameter tuning
- Advanced model evaluation
- Recommendation systems
- Professional visualization

## 📚 Additional Resources

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Python for Data Analysis" by Wes McKinney
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- "Recommender Systems" by Charu C. Aggarwal

## 🤝 Contributing

Feel free to:
- Report bugs and issues
- Suggest improvements
- Add new features
- Enhance visualizations
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: These implementations are designed for educational and demonstration purposes. For production use, additional testing, validation, and optimization may be required.

**Target Accuracy**: High accuracy with comprehensive error handling and validation to ensure reliable results.
