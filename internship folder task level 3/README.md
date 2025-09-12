# Level 3 Machine Learning Tasks

This folder contains advanced machine learning tasks that demonstrate deep learning, audio processing, time series analysis, and computer vision techniques.

## Tasks Overview

### Task 6: Music Genre Classification
**File:** `Task6_Music_Genre_Classification/music_genre_classification.py`  
**Dataset:** `Task6_Music_Genre_Classification/gtzan_dataset.csv`

**Objective:** Classify music genres using audio features and deep learning.

**Key Features:**
- Audio feature extraction (MFCC, Spectral Centroid, Spectral Rolloff, Zero Crossing Rate, Tempo)
- Tabular model training (Random Forest, SVM, Logistic Regression)
- Custom CNN model for spectrogram classification
- Transfer learning with pre-trained models
- Comprehensive visualization and evaluation

**Results:**
- Best Tabular Model: SVM (F1-Score: 0.593)
- Transfer Learning CNN: Accuracy 0.225
- Custom CNN: Accuracy 0.200

---

### Task 7: Sales Forecasting
**File:** `Task7_Sales_Forecasting/sales_forecasting.py`  
**Dataset:** `Task7_Sales_Forecasting/walmart_sales_dataset.csv`

**Objective:** Predict future sales using time series analysis and regression models.

**Key Features:**
- Time-based feature engineering (lag features, rolling averages, seasonal features)
- Multiple regression models (Linear Regression, Random Forest, XGBoost, LightGBM)
- Seasonal decomposition analysis
- Stationarity testing (Augmented Dickey-Fuller Test)
- Future sales forecasting

**Results:**
- Best Model: Linear Regression (R²: 1.000)
- LightGBM: R² 0.471, MAPE 6.86%
- Generated 12-week sales forecast

---

### Task 8: Traffic Sign Recognition
**File:** `Task8_Traffic_Sign_Recognition/traffic_sign_recognition.py`  
**Dataset:** `Task8_Traffic_Sign_Recognition/gtsrb_dataset.csv`

**Objective:** Classify traffic signs using deep learning and computer vision.

**Key Features:**
- 43 traffic sign classes (GTSRB dataset simulation)
- Traditional ML models (Random Forest, SVM, Logistic Regression)
- Custom CNN architecture optimized for small images
- Transfer learning with VGG16
- Data augmentation techniques
- Comprehensive model evaluation

**Results:**
- Best Model: Random Forest (Accuracy: 0.085)
- Custom CNN: Accuracy 0.055
- Transfer Learning: Accuracy 0.075

## Technical Highlights

### Deep Learning Techniques
- **Convolutional Neural Networks (CNNs):** Custom architectures for image classification
- **Transfer Learning:** Pre-trained models (VGG16) for improved performance
- **Data Augmentation:** Rotation, shifting, and zooming for better generalization
- **Batch Normalization:** Improved training stability and convergence

### Audio Processing
- **Feature Extraction:** MFCC, spectral features, tempo analysis
- **Spectrogram Generation:** Time-frequency representation of audio signals
- **Multi-modal Learning:** Combining tabular features with image-based CNNs

### Time Series Analysis
- **Feature Engineering:** Lag features, rolling averages, seasonal decomposition
- **Stationarity Testing:** Statistical tests for time series properties
- **Advanced Models:** XGBoost, LightGBM for time series forecasting
- **Validation:** Time series cross-validation for proper evaluation

### Computer Vision
- **Image Preprocessing:** Normalization, augmentation, resizing
- **Architecture Design:** Optimized CNNs for small image sizes
- **Multi-class Classification:** Handling 43 traffic sign classes
- **Visualization:** Confusion matrices, training curves, sample images

## Libraries Used

### Core ML Libraries
- **scikit-learn:** Traditional machine learning algorithms
- **TensorFlow/Keras:** Deep learning framework
- **XGBoost:** Gradient boosting for regression
- **LightGBM:** Fast gradient boosting

### Specialized Libraries
- **librosa:** Audio feature extraction
- **statsmodels:** Time series analysis and decomposition
- **imbalanced-learn:** Handling class imbalance
- **OpenCV:** Computer vision operations

### Visualization
- **matplotlib:** Static plotting
- **seaborn:** Statistical visualizations
- **Plotly:** Interactive visualizations

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install tensorflow keras xgboost lightgbm
pip install librosa statsmodels imbalanced-learn
pip install opencv-python
```

### Running Individual Tasks
```bash
# Task 6: Music Genre Classification
python "Task6_Music_Genre_Classification/music_genre_classification.py"

# Task 7: Sales Forecasting
python "Task7_Sales_Forecasting/sales_forecasting.py"

# Task 8: Traffic Sign Recognition
python "Task8_Traffic_Sign_Recognition/traffic_sign_recognition.py"
```

## Key Learning Outcomes

### Advanced Machine Learning Concepts
1. **Deep Learning:** CNN architectures, transfer learning, data augmentation
2. **Audio Processing:** Feature extraction, spectrogram analysis
3. **Time Series:** Feature engineering, seasonal decomposition, forecasting
4. **Computer Vision:** Image classification, multi-class problems

### Model Evaluation and Optimization
1. **Cross-validation:** Proper evaluation techniques for different data types
2. **Hyperparameter Tuning:** GridSearchCV for optimal parameters
3. **Performance Metrics:** Accuracy, F1-score, R², MAPE, confusion matrices
4. **Visualization:** Comprehensive plotting for model interpretation

### Data Engineering
1. **Feature Engineering:** Creating meaningful features from raw data
2. **Data Preprocessing:** Normalization, encoding, handling missing values
3. **Data Augmentation:** Techniques to increase dataset size and diversity
4. **Synthetic Data:** Generating realistic datasets for demonstration

## Challenges Addressed

### Technical Challenges
- **Small Dataset:** Limited samples per class in traffic sign recognition
- **High Dimensionality:** Audio features and image pixels
- **Temporal Dependencies:** Time series forecasting with proper validation
- **Class Imbalance:** Handling uneven class distributions

### Model Complexity
- **Overfitting:** Regularization techniques and data augmentation
- **Architecture Design:** Optimizing CNNs for small images
- **Transfer Learning:** Adapting pre-trained models to new domains
- **Ensemble Methods:** Combining multiple models for better performance

## Future Enhancements

### Potential Improvements
1. **Data Collection:** Using real datasets instead of synthetic data
2. **Model Architecture:** Experimenting with newer architectures (ResNet, EfficientNet)
3. **Hyperparameter Optimization:** Automated tuning with Optuna or Ray Tune
4. **Deployment:** Model serving with TensorFlow Serving or FastAPI

### Advanced Techniques
1. **Attention Mechanisms:** Transformer-based models for sequence data
2. **Generative Models:** GANs for data augmentation
3. **Multi-task Learning:** Joint training on related tasks
4. **Federated Learning:** Distributed training across multiple devices

## Conclusion

Level 3 tasks demonstrate advanced machine learning techniques across multiple domains:
- **Audio Processing:** Music genre classification with deep learning
- **Time Series:** Sales forecasting with advanced feature engineering
- **Computer Vision:** Traffic sign recognition with CNNs

These tasks showcase the complexity and power of modern machine learning approaches, providing hands-on experience with state-of-the-art techniques and real-world applications.

---

**Note:** All datasets are synthetically generated for demonstration purposes. In production environments, real datasets should be used for training and evaluation.
