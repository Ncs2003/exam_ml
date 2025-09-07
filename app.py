from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
from typing import List

# Charger le modèle et le scaler
model = joblib.load("log_reg_model.joblib")
scaler = joblib.load("scaler_model.joblib")

expected_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

app = FastAPI(title="API Régression Logistique avec Scaler + CSV")

# CORS pour autoriser Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if not all(col in df.columns for col in expected_columns):
        return {"error": f"Le CSV doit contenir exactement ces colonnes : {expected_columns}"}

    X = df[expected_columns]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    results = [int(p) for p in predictions]
    return {"predictions": results}

@app.post("/predict")
async def predict(data: List[dict]):
    df = pd.DataFrame(data)
    if not all(col in df.columns for col in expected_columns):
        return {"error": f"Les données doivent contenir exactement ces colonnes : {expected_columns}"}

    X = df[expected_columns]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    results = [int(p) for p in predictions]
    return {"predictions": results}
