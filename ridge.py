from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd


DATA_PATH = "result/dataset_cantons_2023_selected_normalized.csv" 
df = pd.read_csv(DATA_PATH)

# Select features and target

TARGET = "loyer_moyen_m2"
FEATURES = [
    "densite_population",
    "taux_logements_vacants",
    "nombre_logements_total",
    "taux_chomage",
    "taux_logements_proprietaire",
]

X = df[FEATURES]
y = df[TARGET]

# Replace NaN values with mean 
imputer = SimpleImputer(strategy="mean") 
X_imputed = imputer.fit_transform(X)

# Normalize features (z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)  # Alpha controls regularization strength

# Cross-validation with MSE and R² (5 folds)
mse_scores_ridge = cross_val_score(ridge_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
r2_scores_ridge = cross_val_score(ridge_model, X_scaled, y, cv=5, scoring='r2')

# Calculate mean performance (MSE and R²)
mean_mse_ridge = -mse_scores_ridge.mean()  # Because cross_val_score returns negative values for MSE
mean_r2_ridge = r2_scores_ridge.mean()

# Display results
print(f"Cross-validated mean MSE for Ridge regression: {mean_mse_ridge}")
print(f"Cross-validated mean R² for Ridge regression: {mean_r2_ridge}")

# Display MSE scores for each fold
print("\nMSE scores for each cross-validation fold (Ridge):")
for i, score in enumerate(mse_scores_ridge, 1):
    print(f"Fold {i}: {abs(score)}")

# Display R² scores for each fold
print("\nR² scores for each cross-validation fold (Ridge):")
for i, score in enumerate(r2_scores_ridge, 1):
    print(f"Fold {i}: {score}")
