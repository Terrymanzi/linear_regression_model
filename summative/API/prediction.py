from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from joblib import load
import numpy as np

# Load assets
model = load("API/best_model.joblib")
scaler = load("API/scaler.joblib")
encoder = load("API/label_encoder.joblib")

# Define app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input validation
class PredictionInput(BaseModel):
    country: str = Field(..., example="Rwanda")
    year: int = Field(..., ge=2000, le=2035, example=2029)

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        encoded_country = encoder.transform([input_data.country])[0]
        input_scaled = scaler.transform([[encoded_country, input_data.year]])
        prediction = model.predict(input_scaled)[0]
        return {
            "predicted_import": round(prediction[0], 2),
            "predicted_export": round(prediction[1], 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
