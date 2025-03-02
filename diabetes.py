import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# Set all random seeds to ensure reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Define features (X) and target variable (y)
X = data.drop('Outcome', axis=1)  # Remove 'Outcome' column from features
y = data['Outcome']  # Target variable

# Replace zero values with NaN for specific columns where 0 is not a valid value
cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[cols_to_replace] = X[cols_to_replace].replace(0, pd.NA)

# Fill NaN values using the median of each column
X.fillna(X.median(), inplace=True)

#  Split the dataset into training and testing sets (80% training, 20% testing)
# Ensure consistent results using random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **Verify data consistency by printing hash values**
print("X_train hash:", hash(str(X_train)))
print("y_train hash:", hash(str(y_train)))

# Standardize features to ensure uniform scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training data and transform
X_test = scaler.transform(X_test)  # Transform test data using the same scaler

# Define the best hyperparameters for the XGBoost model
best_params = {
    'colsample_bytree': 0.9,
    'gamma': 0.1,  # Optimal setting
    'learning_rate': 0.07,
    'max_depth': 6,
    'n_estimators': 400,
    'reg_lambda': 1,
    'scale_pos_weight': 2,  # Balance class distribution
    'subsample': 0.8,
    'eval_metric': "logloss"
}

# Train the XGBoost model using a single CPU core
best_xgb = XGBClassifier(**best_params, random_state=42, tree_method='hist', n_jobs=1)

# Fit the model on training data and evaluate on the test set
best_xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # Monitor performance on test set
    verbose=True
)

# Make predictions
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]  # Get probability scores
threshold = 0.6  # Define a decision threshold
y_pred_xgb = (y_prob_xgb > threshold).astype(int)  # Convert probabilities to class labels

# Evaluate the model performance
print("\n📊 Final Model Performance (Threshold = 0.6):")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))

# Generate a detailed classification report
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Visualize the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Threshold = 0.6)")
plt.show()
