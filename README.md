# House Price Prediction API

Production-ready FastAPI service that wraps a pre-trained Random Forest regressor for house price prediction. Accepts structured JSON input, validates it with Pydantic, and returns a predicted price.

**Portfolio write-up:** [From Notebook to Network: Building a House Price Prediction API](https://viet-in-tech.github.io/house-price-api.html)

**TripleTen Data Science Program · Sprint 11 — Model Deployment**

---

## What's in This Repo

| File | Description |
|------|-------------|
| `app.py` | FastAPI application |
| `model/` | Serialized Random Forest model and preprocessing pipeline |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition |

## How to Run

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Docker

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

## API

**`POST /predict`**

Request body:
```json
{
  "area": 1200,
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 1,
  "parking": 1
}
```

Response:
```json
{
  "predicted_price": 245000.0
}
```

Visit `/docs` (Swagger UI) or `/redoc` (ReDoc) for full schema documentation.

## Tech Stack

- Python · FastAPI · Pydantic
- scikit-learn (RandomForestRegressor)
- Uvicorn · Docker

## Project Status

✅ Approved — TripleTen Data Science Program, Sprint 11
