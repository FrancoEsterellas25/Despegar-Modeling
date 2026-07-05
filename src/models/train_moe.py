import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

# Ajustar path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

def asignar_segmento(df):
    """
    Reglas de negocio heurísticas estáticas (Hard-Routing).
    """
    df_out = df.copy()
    df_out['segmento'] = "Estandar"
    
    # Condición 1: Lujo o Mega Resort
    mask_lujo = (df_out['starRating'] >= 4.5) | (df_out['numberOfRooms'] > 300) | (df_out['total_amenities'] > 15)
    df_out.loc[mask_lujo, 'segmento'] = "Lujo_Resort"
    
    # Condición 2: Boutique / Informal
    mask_boutique = (df_out['segmento'] == "Estandar") & ((df_out['starRating'] < 2) | (df_out['numberOfRooms'] < 50))
    df_out.loc[mask_boutique, 'segmento'] = "Boutique_Informal"
    
    return df_out

def main():
    print("Cargando datos limpios...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.VAL_DATA_PATH)
    
    # Variables a eliminar
    target_col = 'price_by_night_person'
    cols_to_drop = [
        'searchid', 'date', 'name', 'detail', 'destination_name', 'destination_code', 
        'hid', 'geo_id', 'main_city_oid', 'date_search', 'date_ci',
        'price_by_night', 'price_by_night_adult', 'target', 'min_query_price', 'hotel_id_extra'
    ]
    
    print("Separando variables y eliminando columnas de 'leakage'...")
    X_train_full = df_train.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_train.columns], errors='ignore')
    y_train_full = df_train[target_col]
    
    X_val_full = df_val.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_val.columns], errors='ignore')
    y_val_full = df_val[target_col]
    
    # Imputación (replicando Stacking.py para igualdad de condiciones)
    print("Imputando valores faltantes con estadísticas puras del set de Train...")
    num_cols = X_train_full.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns
    
    for col in num_cols:
        mean_val = X_train_full[col].mean()
        X_train_full[col] = X_train_full[col].fillna(mean_val)
        X_val_full[col] = X_val_full[col].fillna(mean_val)
        
    for col in cat_cols:
        if X_train_full[col].isnull().all():
            mode_val = "Missing"
        else:
            mode_val = X_train_full[col].mode()[0]
        X_train_full[col] = X_train_full[col].fillna(mode_val)
        X_val_full[col] = X_val_full[col].fillna(mode_val)
        
    X_train_full = pd.get_dummies(X_train_full, columns=cat_cols, drop_first=True)
    X_val_full = pd.get_dummies(X_val_full, columns=cat_cols, drop_first=True)
    
    # Asegurar mismas columnas
    cols = X_train_full.columns
    for c in cols:
        if c not in X_val_full:
            X_val_full[c] = 0
    X_val_full = X_val_full[cols]
    
    # Retomar Target para el Enrutador
    X_train_full['TARGET'] = y_train_full
    X_val_full['TARGET'] = y_val_full
    
    # Asignar Segmentos
    X_train_routed = asignar_segmento(X_train_full)
    X_val_routed = asignar_segmento(X_val_full)
    
    segmentos = ["Estandar", "Lujo_Resort", "Boutique_Informal"]
    modelos = {}
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    for seg in segmentos:
        print(f"\n--- Entrenando Experto: {seg} ---")
        train_sub = X_train_routed[X_train_routed['segmento'] == seg]
        val_sub = X_val_routed[X_val_routed['segmento'] == seg]
        
        y_t = train_sub['TARGET'].values
        y_v = val_sub['TARGET'].values
        
        # Transformación logarítmica para lidiar con outliers, tal cual el script original
        y_t_log = np.log1p(y_t)
        y_v_log = np.log1p(y_v)
        
        X_t = train_sub.drop(columns=['TARGET', 'segmento']).values
        X_v = val_sub.drop(columns=['TARGET', 'segmento']).values
        
        # Hyperparametros equivalentes al R script
        xgb = XGBRegressor(
            n_estimators=1000,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=30,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1
        )
        
        xgb.fit(
            X_t, y_t_log,
            eval_set=[(X_t, y_t_log), (X_v, y_v_log)],
            verbose=100
        )
        
        modelos[seg] = xgb
    
    print("\nGuardando modelos unificados del MoE...")
    joblib.dump({'modelos': modelos, 'columnas': list(cols)}, os.path.join(config.MODELS_DIR, "moe_model_best.joblib"))
    print("¡Entrenamiento Completo!")

if __name__ == "__main__":
    main()
