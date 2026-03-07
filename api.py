import joblib
import numpy as np
import json
from schemas import HealthCheckResponse

model    = None
metadata = None


def load_model_and_metadata():
    global model, metadata
    try:
        model = joblib.load('model.pkl')
        with open('model_metadata.json') as f:
            metadata = json.load(f)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def make_prediction(house_features):
    feature_values = [house_features[f] for f in metadata['features']]
    X = np.array(feature_values).reshape(1, 13)
    return round(float(model.predict(X)[0]), 2)


def check_health():
    model_ok    = model    is not None
    metadata_ok = metadata is not None
    return HealthCheckResponse(
        status         = "healthy" if model_ok and metadata_ok else "unhealthy",
        model_loaded   = model_ok,
        message        = "Both model and metadata are loaded"
                         if model_ok and metadata_ok
                         else "Model or metadata not loaded"
    )


def get_model_info():
    if metadata:
        return {"version": metadata.get("version", "1.0.0")}
    return {"version": "unknown"}
