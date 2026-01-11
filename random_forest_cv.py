import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "result/dataset_cantons_2023_selected_normalized.csv"
TARGET = "loyer_moyen_m2"

RANDOM_STATE = 42
N_SPLITS = 5

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["canton", TARGET])
y = df[TARGET]

# Pipeline with imputation and Random Forest
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

# Cross-validation
kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# MSE (negative in sklearn)
mse_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=kf,
    scoring="neg_mean_squared_error",
)

mse_scores = np.abs(mse_scores)
mse_mean = mse_scores.mean()

# R²
r2_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=kf,
    scoring="r2",
)

r2_mean = r2_scores.mean()

# Results
print(
    "Cross-validated mean MSE for Random Forest:",
    mse_mean,
)
print(
    "Cross-validated mean R² for Random Forest:",
    r2_mean,
)

print("\nMSE scores for each fold (Random Forest):")
for i, score in enumerate(mse_scores, start=1):
    print(f"Fold {i}: {score}")

print("\nR² scores for each fold (Random Forest):")
for i, score in enumerate(r2_scores, start=1):
    print(f"Fold {i}: {score}")
