from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
from typing import List
import warnings

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if model is None or scaler is None:
        return {"error": "Modèles non chargés"}
    
    try:
        contents = await file.read()
        if not contents.strip():
            return {"error": "Le fichier est vide"}

        df = None
        for enc in ["utf-8", "latin1"]:
            for sep in [",", ";"]:
                try:
                    df = pd.read_csv(io.StringIO(contents.decode(enc)), sep=sep)
                    if not df.empty and all(col in df.columns for col in expected_columns):
                        break
                except Exception:
                    continue
            if df is not None and not df.empty:
                break

        if df is None or df.empty:
            return {"error": "Impossible de lire le CSV. Vérifiez séparateur et encodage."}

        if not all(col in df.columns for col in expected_columns):
            return {"error": f"Le CSV doit contenir exactement ces colonnes : {expected_columns}"}

        X = df[expected_columns]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        results = [int(p) for p in predictions]
        return {"predictions": results}
    
    except Exception as e:
        return {"error": f"Erreur: {str(e)}"}
