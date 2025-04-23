# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel

# Load your processed dataset
df = pd.read_csv('/Users/kshitiz/Desktop/DAV/avian_disease_processed.csv')  # Update path if needed

# Define features and target
X = df.drop(columns=['case_count'])
y = df['case_count']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr = LinearRegression()
ridge = Ridge(alpha=1.0)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train and predict
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation metrics
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}:\n  RMSE: {rmse:.2f}\n  MAE: {mae:.2f}\n  R²: {r2:.3f}\n")
    return rmse, mae, r2

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Ridge Regression", y_test, y_pred_ridge)
evaluate_model("Random Forest", y_test, y_pred_rf)

# Residual plots
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residual Plot: {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.show()

plot_residuals(y_test, y_pred_lr, "Linear Regression")
plot_residuals(y_test, y_pred_ridge, "Ridge Regression")
plot_residuals(y_test, y_pred_rf, "Random Forest")

# Feature importance (Random Forest)
importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:15], y=feature_names[indices][:15])
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Statistical test (Paired t-test between LR and RF)
lr_errors = np.abs(y_test - y_pred_lr)
rf_errors = np.abs(y_test - y_pred_rf)

t_stat, p_value = ttest_rel(lr_errors, rf_errors)
print(f"Paired t-test between Linear Regression and Random Forest:\n  t = {t_stat:.2f}, p = {p_value:.3f}")
