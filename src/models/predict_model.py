import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

def evaluate_predictions(y_true, y_pred, model_name="Modelo"):
    """
    Calcula e imprime pMAE, nRMSE y R2 de forma robusta.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 1. pMAE (Percentage Mean Absolute Error)
    # Expresa el error medio absoluto como porcentaje del promedio real.
    p_mae = (mae / np.mean(y_true)) * 100
    
    # 2. nRMSE (Normalized Root Mean Squared Error)
    # Se reportan dos normalizaciones comunes para mayor rigor académico/profesional:
    # A) Normalización por la media (CV-RMSE)
    n_rmse_mean = (rmse / np.mean(y_true)) * 100
    # B) Normalización por el rango (Max-Min)
    n_rmse_range = (rmse / (np.max(y_true) - np.min(y_true))) * 100
    
    print(f"\n==========================================")
    print(f"MÉTRICAS DE EVALUACIÓN: {model_name}")
    print(f"==========================================")
    print(f"  * MAE:                     {mae:.4f}")
    print(f"  * RMSE:                    {rmse:.4f}")
    print(f"  * pMAE:                    {p_mae:.2f}% (MAE / media_real)")
    print(f"  * nRMSE (por Media):       {n_rmse_mean:.2f}% (RMSE / media_real)")
    print(f"  * nRMSE (por Rango Max-Min): {n_rmse_range:.2f}% (RMSE / rango)")
    print(f"  * R2 Score:                {r2:.4f}")
    
    return p_mae, n_rmse_mean, r2

def main():
    # Obtener el directorio raíz del proyecto (3 niveles arriba desde src/models/predict_model.py)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    # Cargar datos de validación desde la carpeta procesada
    val_path = os.path.join(data_dir, "processed", "val_data_clean.csv")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"No se encontró el dataset de validación en {val_path}")
        
    print(f"Cargando dataset de validación desde: {val_path}")
    val_data = pd.read_csv(val_path)
    
    # Definir características (features)
    features = [
        "position", "adults", "children", "infants", "duration", "anticipation",
        "starRating", "avgRating", "ratio_expectativa",
        "avgRatingCleaning", "avgRatingInternetAccessAndQuality", "avgRatingLocation",
        "avgQualityprice", "avgServicepersonal", "avgService",
        "numberOfRooms", "total_amenities",
        "wday_ci_sine", "wday_ci_cosine",
        "is_carnaval_2024", "is_rock_in_rio_2024", "is_reveillon_2024", "is_carnaval_2025",
        "dist_gig_m", "dist_sdu_m", "dist_metro_m", "dist_favela_m",
        "favela_cercana_size", "favela_densidad_viviendas", "favela_vulnerabilidad_salud",
        "dist_cristo_m", "dist_playa_m", "density_kde",
        "dist_mean_5nn", "dist_mean_10nn", "dist_mean_20nn"
    ]
    for i in range(1, 9):
        features.append(f"mca_dim_{i}")
        
    X_val = val_data[features]
    y_val = val_data["price_by_night_person"]
    
    # Cargar y evaluar el modelo Raw (Escenario 5) si existe
    raw_model_path = os.path.join(models_dir, "lgb_model_raw.txt")
    if os.path.exists(raw_model_path):
        print(f"Cargando modelo Raw desde: {raw_model_path}")
        lgb_model_raw = lgb.Booster(model_file=raw_model_path)
        y_pred_raw = lgb_model_raw.predict(X_val)
        evaluate_predictions(y_val, y_pred_raw, model_name="LightGBM (Target Raw)")
    else:
        print(f"Aviso: No se encontró el modelo Raw en {raw_model_path}")
        
    # Cargar y evaluar el modelo Logarítmico (Escenario 6) si existe
    log_model_path = os.path.join(models_dir, "lgb_model_log.txt")
    if os.path.exists(log_model_path):
        print(f"Cargando modelo Logarítmico desde: {log_model_path}")
        lgb_model_log = lgb.Booster(model_file=log_model_path)
        y_pred_log_scaled = lgb_model_log.predict(X_val)
        y_pred_log = np.exp(y_pred_log_scaled)
        evaluate_predictions(y_val, y_pred_log, model_name="LightGBM (Target Log)")
    else:
        print(f"Aviso: No se encontró el modelo Log en {log_model_path}")

if __name__ == "__main__":
    main()
