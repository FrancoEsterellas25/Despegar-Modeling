import os
import json
import numpy as np
import polars as pl
from src.models.Stacking import load_and_clean_data
import config
from src.services.model_service import ModelService

def calc_metrics(y, p):
    y_mean = np.mean(y) if np.mean(y) != 0 else 1e-6
    y_range = np.max(y) - np.min(y)
    if y_range == 0: y_range = 1e-6
    
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    
    n_rmse = rmse / y_range
    p_mae = (mae / y_mean) * 100.0
    r2 = 1 - (np.sum((y - p) ** 2) / np.sum((y - y_mean) ** 2))
    return rmse, n_rmse, mae, p_mae, r2

def main():
    print("Cargando datos de validación...", flush=True)
    _, _, X_val_df, y_val_series = load_and_clean_data(config.TRAIN_DATA_PATH, config.VAL_DATA_PATH)
    features = pl.DataFrame(X_val_df)
    y_true = y_val_series.to_numpy()
    
    print("Inicializando ModelService...", flush=True)
    model_service = ModelService(use_onnx=True)
    
    print("Generando predicciones para Stacking...", flush=True)
    preds_stacking = model_service.predict("Stacking", features)
    
    print("Generando predicciones para MoE...", flush=True)
    preds_moe = model_service.predict("MoE", features)
    
    print("Calculando métricas...", flush=True)
    metrics = []
    
    # Stacking
    r_stacking, nr_stacking, mae_stacking, pm_stacking, r2_stacking = calc_metrics(y_true, preds_stacking)
    metrics.append({
        "Arquitectura": "Ensamble Stacking (Meta-Nivel)",
        "MAE": mae_stacking,
        "RMSE": r_stacking,
        "n-RMSE": nr_stacking,
        "p-MAE (%)": pm_stacking,
        "R²": r2_stacking
    })
    
    # MoE (Métricas Reales Calculadas sobre Validation Set con XGBoost Python)
    r_moe, nr_moe, mae_moe, pm_moe, r2_moe = calc_metrics(y_true, preds_moe)
    
    metrics.append({
        "Arquitectura": "Mixture of Experts (Enrutador Python)",
        "MAE": mae_moe,
        "RMSE": r_moe,
        "n-RMSE": nr_moe,
        "p-MAE (%)": pm_moe,
        "R²": r2_moe
    })
    
    os.makedirs("data/cache", exist_ok=True)
    
    print("Guardando métricas y predicciones...", flush=True)
    with open("data/cache/metrics_cache.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    np.savez_compressed("data/cache/preds_cache.npz", y_true=y_true, preds_stacking=preds_stacking, preds_moe=preds_moe)
    features.write_parquet("data/cache/val_features_processed.parquet")
    print("Proceso completado con éxito.", flush=True)

if __name__ == "__main__":
    main()
