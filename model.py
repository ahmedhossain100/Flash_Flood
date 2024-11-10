import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, precision_recall_curve

# Load the data
data = pd.read_excel('output_statistics.xlsx')

# Convert 'Date' to datetime and extract month
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month

# One-hot encoding for the 'Month' using pandas get_dummies
data = pd.concat([data, pd.get_dummies(data['Month'], prefix='Month', drop_first=True)], axis=1)

# Prepare features and target
X = data.drop(['Date', 'Rainfall', 'Extreme Rainfall', 'Month'], axis=1)  # Exclude non-feature columns
y = (data['Rainfall'] > 204.5).astype(int)  # Binary target for high rainfall

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), X_train.columns)  # Scale all features
    ]
)

# Create the LightGBM model
model = LGBMClassifier(class_weight='balanced', random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Parameter grid for GridSearchCV
param_grid = {
    'classifier__num_leaves': [31, 50],
    'classifier__max_depth': [10, 20, 30],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [100, 200]
}

# Grid search focusing on recall
grid_search = GridSearchCV(pipeline, param_grid, scoring='recall', n_jobs=-1, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

# Save the best model to a pickle file for deployment
with open('best_model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

# Predict probabilities for the test set
probabilities = grid_search.predict_proba(X_test)[:, 1]

# Calculate precision-recall pairs for different probability thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)

# Find the threshold that provides the best balance between precision and recall
best_threshold = thresholds[np.argmax(precisions >= 0.20)]  # Example for minimum acceptable precision

# Apply the best threshold to make final predictions
final_predictions = (probabilities >= best_threshold).astype(int)

# Recalculate metrics with the new threshold
final_accuracy = accuracy_score(y_test, final_predictions)
final_recall = recall_score(y_test, final_predictions)
final_precision = precision_score(y_test, final_predictions)
final_roc_auc = roc_auc_score(y_test, probabilities)  # ROC AUC remains the same as it's threshold invariant

print(f"Best parameters: {grid_search.best_params_}")
print(f"Final Accuracy: {final_accuracy:.2f}")
print(f"Final Recall: {final_recall:.2f}")
print(f"Final Precision: {final_precision:.2f}")
print(f"Final ROC AUC Score: {final_roc_auc:.2f}")
