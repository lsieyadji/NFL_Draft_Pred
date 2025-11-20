# random_forest_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os

# Set up paths
data_dir = r'C:\Users\cash\coding_projects\NFL_Draft_Pred\data'
models_dir = r'C:\Users\cash\coding_projects\NFL_Draft_Pred\models'
os.makedirs(models_dir, exist_ok=True)

# Load cleaned data
print("Loading cleaned data...")
df = pd.read_csv(os.path.join(data_dir, 'NFL_cleaned.csv'))
print(f"Data shape: {df.shape}")

# Prepare features and target
X = df.drop('Drafted', axis=1)
y = df['Drafted']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*50)
print("RANDOM FOREST PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save the model
model_path = os.path.join(models_dir, 'random_forest_model.pkl')
joblib.dump(rf_model, model_path)
print(f"\nModel saved to: {model_path}")

print("\nRandom Forest training complete! ✅")