import os
import json
import sys
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Amazon Green Electronics Pricing API",
    description="Backend API for Dynamic Model Predictions (OLS, Ridge, RandomForest, XGBoost)",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force utf-8 stdout encoding if possible
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 1. LOAD MODELS & METRICS AT STARTUP
# -----------------------------------------------------------------------------
MODELS_DIR = "models"
models = {}
metrics = {}

# Map request parameter to saved model filename
MODEL_MAPPING = {
    "ols": "ols_model.joblib",
    "ridge": "ridge_model.joblib",
    "rf": "rf_model.joblib",
    "xgboost": "xgb_model.joblib"
}

# Display names in metrics.json
METRICS_MAPPING = {
    "ols": "OLS Regression",
    "ridge": "Ridge Regression",
    "rf": "Random Forest",
    "xgboost": "XGBoost"
}

@app.on_event("startup")
def load_resources():
    print("=" * 60)
    print("Initializing API Server & Loading ML Models...")
    print("=" * 60)
    
    # Load metrics.json
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                global metrics
                metrics = json.load(f)
            print("Loaded metrics.json successfully.")
        except Exception as e:
            print(f"Error loading metrics.json: {e}")
    else:
        print("metrics.json not found. Metrics fallback will be empty.")

    # Load all models
    for key, filename in MODEL_MAPPING.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            try:
                models[key] = joblib.load(model_path)
                print(f"Loaded Model '{key}' from '{model_path}'")
            except Exception as e:
                print(f"Failed to load Model '{key}' from '{model_path}': {e}")
        else:
            print(f"Model file '{model_path}' not found. Please run train_models.py first.")
            
    print("=" * 60)

# -----------------------------------------------------------------------------
# 2. SCHEMAS & UTILITIES
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Model key: 'ols', 'ridge', 'rf', or 'xgboost'")
    price: float = Field(..., gt=0, description="Product price in USD")
    is_green: int = Field(..., ge=0, le=1, description="Green sustainability flag (0 or 1)")
    rating: float = Field(..., ge=1.0, le=5.0, description="Product rating (1.0 to 5.0)")
    reviews_count: int = Field(..., ge=0, description="Number of reviews")

class CompareRequest(BaseModel):
    price: float = Field(..., gt=0, description="Product price in USD")
    is_green: int = Field(..., ge=0, le=1, description="Green sustainability flag (0 or 1)")
    rating: float = Field(..., ge=1.0, le=5.0, description="Product rating (1.0 to 5.0)")
    reviews_count: int = Field(..., ge=0, description="Number of reviews")

def perform_prediction(model_key: str, price: float, is_green: int, rating: float, reviews_count: int) -> float:
    """Helper to perform feature prep and run model prediction."""
    if model_key not in models:
        raise HTTPException(
            status_code=503, 
            detail=f"Model '{model_key}' is not loaded/available on server."
        )
    
    # 1. Feature Engineering (matching training pipeline)
    log_price = np.log1p(price)
    log_price_x_is_green = log_price * is_green
    
    # Structure features into DataFrame matching order of X features:
    # ['log_price', 'is_green', 'log_price_x_is_green', 'rating', 'reviews_count']
    input_data = pd.DataFrame([{
        "log_price": log_price,
        "is_green": float(is_green),
        "log_price_x_is_green": log_price_x_is_green,
        "rating": float(rating),
        "reviews_count": float(reviews_count)
    }])
    
    # 2. Model Prediction (predicts log_sales)
    model = models[model_key]
    pred_log_sales = model.predict(input_data)[0]
    
    # Retransform from log(y+1) to original units: exp(y) - 1
    # Apply standard expm1, clamping to 0 for logical consistency
    pred_sales = float(max(0, np.expm1(pred_log_sales)))
    return pred_sales

def get_model_metrics(model_key: str) -> dict:
    """Helper to retrieve cached model evaluation metrics."""
    json_key = METRICS_MAPPING.get(model_key, "")
    return metrics.get(json_key, {
        "R2_Score": None,
        "RMSE": None,
        "MAE": None
    })

# -----------------------------------------------------------------------------
# 3. ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Amazon Green Electronics Pricing API Server",
        "available_models": list(models.keys())
    }

@app.post("/api/predict")
def predict_sales(payload: PredictRequest):
    model_name = payload.model_name.lower().strip()
    if model_name not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_name '{payload.model_name}'. Allowed options: 'ols', 'ridge', 'rf', 'xgboost'"
        )
        
    try:
        pred_sales = perform_prediction(
            model_name, payload.price, payload.is_green, payload.rating, payload.reviews_count
        )
        
        model_perf = get_model_metrics(model_name)
        
        return {
            "model_name": model_name,
            "predicted_sales": round(pred_sales, 2),
            "metrics": {
                "R2_Score": model_perf.get("R2_Score"),
                "RMSE": model_perf.get("RMSE"),
                "MAE": model_perf.get("MAE")
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/compare")
def compare_models(payload: CompareRequest):
    results = []
    
    # Iterate and predict on all available models
    for model_key in MODEL_MAPPING.keys():
        if model_key not in models:
            continue
        try:
            pred_sales = perform_prediction(
                model_key, payload.price, payload.is_green, payload.rating, payload.reviews_count
            )
            model_perf = get_model_metrics(model_key)
            
            results.append({
                "model_name": model_key,
                "display_name": METRICS_MAPPING.get(model_key, model_key),
                "predicted_sales": round(pred_sales, 2),
                "metrics": {
                    "R2_Score": model_perf.get("R2_Score"),
                    "RMSE": model_perf.get("RMSE"),
                    "MAE": model_perf.get("MAE")
                }
            })
        except Exception as e:
            # Skip erroring models or append empty result
            results.append({
                "model_name": model_key,
                "display_name": METRICS_MAPPING.get(model_key, model_key),
                "predicted_sales": None,
                "error": str(e)
            })
            
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
