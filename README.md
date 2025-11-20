#  Draft Predict Pro

A machine learning project that predicts whether NFL Combine participants will be drafted based on their athletic metrics, physical attributes, and position.

## Project Overview

This project leverages machine learning to analyze NFL Combine data and predict draft selection outcomes. Using ensemble methods and multiple algorithms, the model achieves **70%+ accuracy** in identifying which prospects will be drafted.

##  Features

- Multiple Model Architectures: Random Forest, XGBoost, Voting Ensemble, and Stacking Ensemble
- Web Interface: Interactive Flask web application for real-time predictions
- Feature Analysis: Comprehensive Combine metric evaluation (40-yard dash, vertical jump, bench press, etc.)
- Position-Specific Predictions: Accounts for different athletic requirements by position

## Model Performance

| Model | Accuracy | ROC AUC | Undrafted Recall | Drafted Recall |
|-------|----------|---------|------------------|----------------|
| Random Forest | 70.4% | 0.722 | 36% | 89% |
| XGBoost | 69.3% | 0.701 | 47% | 82% |
| Stacking Ensemble | 70.3% | 0.721 | 40% | 87% |
| Voting Ensemble | 68.7% | 0.714 | 41% | 84% |

**Stacking Ensemble Recommended**: Best balance of accuracy and class prediction balance

##  Installation

1. **Clone the repository**
   ```bash
git clone https://github.com/lsieyadji/NFL_Draft_Pred.git
cd NFL_Draft_Pred
pip install -r requirements.txt
cd website
python app.py
Access the application
Open http://localhost:5000 in your browser

 Project Structure
text
NFL_Draft_Pred/
├── models/                 # Trained machine learning models
├── data/                  # Raw and cleaned datasets
├── src/                   # Data processing and model training scripts
├── website/               # Flask web application
│   ├── app.py            # Flask backend
│   ├── templates/        # HTML templates
│   └── static/           # CSS and JavaScript files
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
## Usage
Web Application
Select player position

Enter Combine metrics:

Height (meters)

Weight (kg)

40-yard dash time

Vertical jump (cm)

Bench press reps

Broad jump (cm)

BMI

Age

Click "Predict Draft Selection" for instant results

## Model Training
Training scripts are available in the src/ directory:

data_cleaning.py - Data cleaning and feature engineering

Random_forest_code.py - Random Forest implementation

Xgboost_code.py - XGBoost implementation

Stacking_code.py - Stacking ensemble model

Ensemble_code.py - Voting ensemble model

 ## Data Sources
NFL Combine data (2009-2012)

Player physical attributes and athletic metrics

Draft results and selection information

 ## Technical Details
Algorithms: Random Forest, XGBoost, Ensemble Methods

Framework: Scikit-learn, XGBoost, Flask

Frontend: HTML5, CSS3, JavaScript

Validation: 5-fold cross-validation, train-test splits

 ## Key Insights
40-yard dash and vertical jump are among the most predictive features

Position-specific thresholds significantly impact draft probability

Class imbalance between drafted/undrafted players presents modeling challenges

Ensemble methods provide the most robust predictions
