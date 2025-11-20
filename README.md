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

## 🛠️ Installation

1. **Clone the repository**
   ```bash
git clone https://github.com/lsieyadji/NFL_Draft_Pred.git
cd NFL_Draft_Pred

```bash
git clone https://github.com/lsieyadji/NFL_Draft_Pred.git
cd NFL_Draft_Pred
