from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
from typing import List
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle et le scaler avec gestion d'erreur
try:
    model = joblib.load("log_reg_model.joblib")
    scaler = joblib.load("scaler_model.joblib")
    logger.info("Modèles chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des modèles : {e}")
    raise

expected_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

app = FastAPI(
    title="API Régression Logistique avec Scaler + CSV",
    description="API pour prédire des billets authentiques/contrefaits",
    version="1.0.0"
)

# CORS pour autoriser Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Point d'entrée principal de l'API"""
    return {
        "message": "API Régression Logistique - Service actif",
        "version": "1.0.0",
        "endpoints": {
            "documentation": "/docs",
            "predict_file": "/predict_file",
            "predict": "/predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de santé de l'API"""
    try:
        # Test simple pour vérifier que les modèles sont accessibles
        test_data = pd.DataFrame([[0.0] * len(expected_columns)], columns=expected_columns)
        scaler.transform(test_data)
        model.predict(scaler.transform(test_data))
        return {
            "status": "healthy",
            "models_loaded": True,
            "expected_columns": expected_columns
        }
    except Exception as e:
        logger.error(f"Erreur lors du health check : {e}")
        raise HTTPException(status_code=500, detail="Service indisponible")

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """Prédiction à partir d'un fichier CSV"""
    try:
        # Vérifier le type de fichier
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Le fichier doit être un CSV")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        logger.info(f"Fichier CSV lu avec {len(df)} lignes")

        # Vérifier les colonnes
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Colonnes manquantes : {missing_columns}. Colonnes attendues : {expected_columns}"
            )

        X = df[expected_columns]
        
        # Vérifier les valeurs manquantes
        if X.isnull().any().any():
            raise HTTPException(status_code=400, detail="Le dataset contient des valeurs manquantes")
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i,
                "prediction": int(pred),
                "probability_class_0": float(prob[0]),
                "probability_class_1": float(prob[1]),
                "confidence": float(max(prob))
            })
        
        logger.info(f"Prédictions effectuées pour {len(results)} échantillons")
        return {
            "predictions": results,
            "total_samples": len(results),
            "filename": file.filename
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Erreur de parsing CSV : {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

@app.post("/predict")
async def predict(data: List[dict]):
    """Prédiction à partir de données JSON"""
    try:
        if not data:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie")
        
        df = pd.DataFrame(data)
        logger.info(f"Données reçues : {len(df)} échantillons")
        
        # Vérifier les colonnes
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Colonnes manquantes : {missing_columns}. Colonnes attendues : {expected_columns}"
            )

        X = df[expected_columns]
        
        # Vérifier les valeurs manquantes
        if X.isnull().any().any():
            raise HTTPException(status_code=400, detail="Les données contiennent des valeurs manquantes")
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i,
                "prediction": int(pred),
                "probability_class_0": float(prob[0]),
                "probability_class_1": float(prob[1]),
                "confidence": float(max(prob))
            })
        
        logger.info(f"Prédictions effectuées pour {len(results)} échantillons")
        return {
            "predictions": results,
            "total_samples": len(results)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

@app.get("/model_info")
async def model_info():
    """Informations sur le modèle"""
    try:
        return {
            "model_type": str(type(model).__name__),
            "scaler_type": str(type(scaler).__name__),
            "expected_columns": expected_columns,
            "n_features": len(expected_columns),
            "classes": model.classes_.tolist() if hasattr(model, 'classes_') else None
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos du modèle : {e}")
        raise HTTPException(status_code=500, detail="Impossible de récupérer les informations du modèle")