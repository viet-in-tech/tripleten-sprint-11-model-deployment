from fastapi import FastAPI, HTTPException
from schemas import HousePredictionRequest, PredictionResponse
from api import load_model_and_metadata, make_prediction, check_health, get_model_info

app = FastAPI(title="House Price Prediction API", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    success = load_model_and_metadata()
    if not success:
        print("WARNING: Failed to load model at startup")


@app.get("/health")
async def health():
    return check_health()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    try:
        features_dict   = request.dict()
        predicted_price = make_prediction(features_dict)
        model_version   = get_model_info()["version"]
        return PredictionResponse(
            predicted_price=predicted_price,
            currency="USD",
            model_version=model_version
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
