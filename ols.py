import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Linear Regression 
ols_model = LinearRegression()

# Cross-validation
mse_scores_ols = cross_val_score(ols_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
r2_scores_ols = cross_val_score(ols_model, X_scaled, y, cv=5, scoring='r2')

# Calculate mean MSE and R²
mean_mse_ols = -mse_scores_ols.mean()
mean_r2_ols = r2_scores_ols.mean()

# Display results
print(f"Cross-validated mean MSE for linear regression: {mean_mse_ols}")
print(f"Cross-validated mean R² for linear regression: {mean_r2_ols}")

print("\nMSE scores for each cross-validation fold (OLS):")
for i, score in enumerate(mse_scores_ols, 1):
    print(f"Fold {i}: {abs(score)}")

print("\nR² scores for each cross-validation fold (OLS):")
for i, score in enumerate(r2_scores_ols, 1):
    print(f"Fold {i}: {score}")
