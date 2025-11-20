# stacking_ensemble.py
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os

# Set up paths
data_dir = r'C:\Users\cash\coding_projects\NFL_Draft_Pred\data'
models_dir = r'C:\Users\cash\coding_projects\NFL_Draft_Pred\models'

# Load data
print("Loading data...")
df = pd.read_csv(os.path.join(data_dir, 'NFL_cleaned.csv'))
X = df.drop('Drafted', axis=1)
y = df['Drafted']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load your trained models
print("Loading pre-trained models...")
rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))

# Create stacking ensemble
print("Creating stacking ensemble...")
stacking_ensemble = StackingClassifier(
    estimators=[
        ('random_forest', rf_model),
        ('xgboost', xgb_model)
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5,  # 5-fold cross-validation for the meta-learner
    passthrough=False  # Use only predictions from base models
)

# Train the stacking ensemble
print("Training stacking ensemble...")
stacking_ensemble.fit(X_train, y_train)

# Evaluate
y_pred = stacking_ensemble.predict(X_test)
y_pred_proba = stacking_ensemble.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("STACKING ENSEMBLE PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save stacking model
joblib.dump(stacking_ensemble, os.path.join(models_dir, 'stacking_ensemble_model.pkl'))
print("\nStacking ensemble model saved!")

# Compare with individual models
print("\n" + "="*50)
print("COMPARISON WITH INDIVIDUAL MODELS")
print("="*50)
print(f"Random Forest: 70.4%")
print(f"XGBoost:      69.3%")
print(f"Voting:       68.7%")
print(f"Stacking:     {accuracy*100:.1f}%")