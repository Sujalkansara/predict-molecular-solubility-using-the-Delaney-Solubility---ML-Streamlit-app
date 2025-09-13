import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset (replace with your CSV path or URL)
data = pd.read_csv('https://raw.githubusercontent.com/Sujalkansara/predict-molecular-solubility-using-the-Delaney-Solubility-ML/refs/heads/main/delaney_solubility_with_descriptors.csv')

# Features and target
X = data[['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion']]
y = data['logS']

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)
print(f"Linear Regression - MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}")

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tune this
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print(f"Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

# Save models
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')

# Optional: Save a sample scaler if you preprocess (not needed here as data is clean)
print("Models saved successfully!")