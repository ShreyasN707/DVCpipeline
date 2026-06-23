from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from fastapi.staticfiles import StaticFiles         # <-- NEW IMPORT
from fastapi.responses import FileResponse



app=FastAPI(
    title="Stellar Classification API",
    description="An ML microservice that classifies Galaxies, Stars, and Quasars based on light signatures.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
# -------------------------

class StellarFeatures(BaseModel):
    redshift: float
    u: float  # <-- NEW
    g: float
    i: float
    r: float
    z: float
    delta: float

try:
    pipeline = joblib.load("models/stellar_pipeline.joblib")
    le = joblib.load("models/label_encoder.joblib")
except Exception as e:
    print(f"[-] FATAL: Could not load artifacts. Error: {e}")

@app.post("/predict")
async def predict_stellar_object(features: StellarFeatures):
    try:
        # Step A: Extract the raw data from the frontend
        data = features.dict()
        
        # Step B: Calculate the physics features (Color Indices) on the fly
        u_g_color = data['u'] - data['g']
        g_r_color = data['g'] - data['r']
        r_i_color = data['r'] - data['i']
        i_z_color = data['i'] - data['z']
        
        # Step C: Build the exact 9-column DataFrame the model was trained on
        # The column names MUST match the list you used in preprocess.py
        df = pd.DataFrame([{
            'redshift': data['redshift'],
            'i': data['i'],
            'r': data['r'],
            'z': data['z'],
            'delta': data['delta'],
            'u_g_color': u_g_color,
            'g_r_color': g_r_color,
            'r_i_color': r_i_color,
            'i_z_color': i_z_color
        }])
        
        # Step D: Predict
        numeric_prediction = pipeline.predict(df)[0]
        text_prediction = le.inverse_transform([numeric_prediction])[0]
        
        return {"status": "success", "prediction": text_prediction}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))