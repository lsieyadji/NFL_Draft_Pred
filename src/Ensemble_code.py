# ensemble_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
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
rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))

# Create ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'  # Uses probability scores for better performance
)

print("Training ensemble model...")
ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("ENSEMBLE PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save ensemble model
joblib.dump(ensemble, os.path.join(models_dir, 'ensemble_model.pkl'))
print("\nEnsemble model saved!")