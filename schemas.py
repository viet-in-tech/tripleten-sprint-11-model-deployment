from pydantic import BaseModel, Field
from datetime import datetime


class HousePredictionRequest(BaseModel):
    total_images:     int   = Field(..., ge=0,    le=50)
    beds:             int   = Field(..., ge=0,    le=10)
    baths:            float = Field(..., ge=0,    le=10)
    area:             float = Field(..., gt=0,    le=10000)
    latitude:         float = Field(..., ge=-90,  le=90)
    longitude:        float = Field(..., ge=-180, le=180)
    garden:           int   = Field(..., ge=0,    le=1)
    garage:           int   = Field(..., ge=0,    le=1)
    new_construction: int   = Field(..., ge=0,    le=1)
    pool:             int   = Field(..., ge=0,    le=1)
    terrace:          int   = Field(..., ge=0,    le=1)
    air_conditioning: int   = Field(..., ge=0,    le=1)
    parking:          int   = Field(..., ge=0,    le=1)


class PredictionResponse(BaseModel):
    predicted_price: float = Field(description="Predicted price in USD")
    currency:        str   = Field(default="USD")
    model_version:   str   = Field(description="Model version string")


class HealthCheckResponse(BaseModel):
    status:       str  = Field(description="healthy or unhealthy")
    model_loaded: bool
    message:      str
