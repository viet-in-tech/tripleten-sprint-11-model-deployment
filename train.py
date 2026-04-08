"""
Train a Random Forest regressor on the house price dataset and serialize
the model and metadata needed by api.py.

Usage:
    python train.py

Outputs:
    model.pkl            — serialized RandomForestRegressor
    model_metadata.json  — feature list, version, and evaluation metrics
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Feature columns — must match the 13 fields in schemas.HousePredictionRequest
# ---------------------------------------------------------------------------
FEATURES = [
    'total_images',
    'beds',
    'baths',
    'area',
    'latitude',
    'longitude',
    'garden',
    'garage',
    'new_construction',
    'pool',
    'terrace',
    'air_conditioning',
    'parking',
]
TARGET = 'price'

MODEL_VERSION = '1.0.0'


def load_data(path: str = 'house_prices.csv') -> pd.DataFrame:
    """Load the house price dataset."""
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Dataset missing columns: {missing}')
    return df[required].dropna()


def train(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    print(f'Test MAE: ${mae:,.2f}')
    print(f'Test R²:  {r2:.4f}')

    return model, {'mae': round(mae, 2), 'r2': round(r2, 4)}


def save_artifacts(model, metrics: dict):
    joblib.dump(model, 'model.pkl')
    print('Saved model.pkl')

    metadata = {
        'version':  MODEL_VERSION,
        'features': FEATURES,
        'target':   TARGET,
        'metrics':  metrics,
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print('Saved model_metadata.json')
    print('Metadata:', json.dumps(metadata, indent=2))


if __name__ == '__main__':
    print('Loading data...')
    df = load_data()
    print(f'Dataset: {len(df):,} rows, {len(FEATURES)} features')

    print('Training model...')
    model, metrics = train(df)

    print('Saving artifacts...')
    save_artifacts(model, metrics)

    print('Done. Run the API with: uvicorn main:app --reload')
