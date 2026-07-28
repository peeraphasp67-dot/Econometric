import os
import json
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Force utf-8 stdout encoding for Windows console compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print("=" * 60)
    print("Starting Model Training & Evaluation Process")
    print("=" * 60)

    # 1. Load Data
    data_path = 'data/amazon_products_sales_data_cleaned.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    
    # Drop rows with missing values in target and core feature columns
    df_clean = df.dropna(subset=['purchased_last_month', 'discounted_price', 'product_rating', 'total_reviews']).copy()
    print(f"Dataset loaded successfully. Total clean samples: {len(df_clean):,}")

    # 2. Feature Engineering
    GREEN_TAGS = ['Carbon impact', 'Energy efficiency', 'Manufacturing practices', 'Forestry practices', 'Recycled materials']
    
    # Create required features matching user specification
    df_clean['log_price'] = np.log1p(df_clean['discounted_price'])
    df_clean['is_green'] = df_clean['sustainability_tags'].isin(GREEN_TAGS).astype(int)
    df_clean['log_price_x_is_green'] = df_clean['log_price'] * df_clean['is_green']
    df_clean['rating'] = df_clean['product_rating']
    df_clean['reviews_count'] = df_clean['total_reviews']
    df_clean['log_sales'] = np.log1p(df_clean['purchased_last_month'])

    # Define Feature Matrix (X) and Target Vector (y)
    feature_cols = ['log_price', 'is_green', 'log_price_x_is_green', 'rating', 'reviews_count']
    X = df_clean[feature_cols]
    y = df_clean['log_sales']

    print("\nFeature Matrix (X) Shape:", X.shape)
    print("Target Vector (y) Shape:", y.shape)

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train-Test Split (80/20): Train N={len(X_train):,}, Test N={len(X_test):,}")

    # 4. Initialize Models
    models = {
        "OLS Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

    # Model filenames mapping
    model_filenames = {
        "OLS Regression": "ols_model.joblib",
        "Ridge Regression": "ridge_model.joblib",
        "Random Forest": "rf_model.joblib",
        "XGBoost": "xgb_model.joblib"
    }

    # 5. Train & Evaluate Models
    metrics_results = {}
    evaluation_rows = []

    print("\nTraining and evaluating models...")
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on Test set
        y_pred = model.predict(X_test)
        
        # Calculate Metrics on Log Scale
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics_results[model_name] = {
            "R2_Score": round(float(r2), 4),
            "RMSE": round(float(rmse), 4),
            "MAE": round(float(mae), 4),
            "Filename": model_filenames[model_name]
        }

        evaluation_rows.append({
            "Model": model_name,
            "R2 Score": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })

    # 6. Display Comparison DataFrame
    eval_df = pd.DataFrame(evaluation_rows)
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON (TEST SET)")
    print("=" * 60)
    print(eval_df.to_string(index=False))
    print("=" * 60)

    # 7. Save Models and Metrics
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving model artifacts and metrics to '{output_dir}/'...")

    for model_name, model in models.items():
        save_path = os.path.join(output_dir, model_filenames[model_name])
        joblib.dump(model, save_path)
        print(f"  Saved {model_name:20s} -> {save_path}")

    # Save metrics JSON
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_results, f, indent=4, ensure_ascii=False)
    print(f"  Saved Evaluation Metrics -> {metrics_json_path}")

    print("\nAll 4 models trained, evaluated, and saved successfully!")

if __name__ == "__main__":
    main()
