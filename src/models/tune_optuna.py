import optuna
import pandas as pd
import numpy as np
import os
import sys
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ajustar path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

def main():
    print("Iniciando Optuna con Optimizacion Bayesiana (TPE)...")
    
    # 1. Cargar Datos
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.VAL_DATA_PATH)
    
    target_col = 'price_by_night_person'
    cols_to_drop = [
        'searchid', 'date', 'name', 'detail', 'destination_name', 'destination_code', 
        'hid', 'geo_id', 'main_city_oid', 'date_search', 'date_ci',
        'price_by_night', 'price_by_night_adult', 'target', 'min_query_price', 'hotel_id_extra'
    ]
    
    X_train = df_train.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_train.columns], errors='ignore')
    y_train = df_train[target_col]
    X_val = df_val.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_val.columns], errors='ignore')
    y_val = df_val[target_col]
    
    # 2. Imputación Rápida
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    for col in num_cols:
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_val[col] = X_val[col].fillna(mean_val)
        
    for col in cat_cols:
        mode_val = "Missing" if X_train[col].isnull().all() else X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_val[col] = X_val[col].fillna(mode_val)
        
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
    
    for c in X_train.columns:
        if c not in X_val:
            X_val[c] = 0
    X_val = X_val[X_train.columns]
    
    # 3. K-Means Routing (Fijado en K=9)
    cluster_feats = [c for c in X_train.columns if 'mca_dim' in c or 'dist_' in c or 'Rating' in c or c == 'numberOfRooms']
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train[cluster_feats].fillna(0))
    X_va_sc = scaler.transform(X_val[cluster_feats].fillna(0))
    
    best_k = 9
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    tr_labels = kmeans.fit_predict(X_tr_sc)
    va_labels = kmeans.predict(X_va_sc)
    
    X_train['segmento'] = [f"Cluster_{i}" for i in tr_labels]
    X_train['TARGET'] = y_train
    
    X_val['segmento'] = [f"Cluster_{i}" for i in va_labels]
    X_val['TARGET'] = y_val
    
    # Para ahorrar muchísimo tiempo en Optuna, vamos a optimizar los hiperparámetros
    # utilizando únicamente el Clúster más grande como "Proxy" representativo.
    segmento_proxy = X_train['segmento'].value_counts().index[0]
    print(f"Optimizando sobre el clúster mayoritario ({segmento_proxy}) como proxy...")
    
    train_sub = X_train[X_train['segmento'] == segmento_proxy]
    val_sub = X_val[X_val['segmento'] == segmento_proxy]
    
    X_t = train_sub.drop(columns=['TARGET', 'segmento']).values
    y_t = train_sub['TARGET'].values
    
    X_v = val_sub.drop(columns=['TARGET', 'segmento']).values
    y_v = val_sub['TARGET'].values

    # 4. Función Objetivo de Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        
        xgb = XGBRegressor(
            **params,
            early_stopping_rounds=20,
            objective="reg:squarederror", 
            eval_metric="rmse", 
            random_state=42, 
            n_jobs=-1
        )
        
        xgb.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=0)
        
        preds = xgb.predict(X_v)
        score = r2_score(y_v, preds)
        
        return score

    # 5. Lanzar Optimización (Maximizando R2)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    print("Iniciando búsqueda de 15 trials (Modo Fast)...")
    study.optimize(objective, n_trials=15)
    
    print("\n" + "="*50)
    print("¡OPTIMIZACIÓN BAYESIANA FINALIZADA!")
    print("="*50)
    print(f"Mejor R2 encontrado: {study.best_value:.4f}")
    print("Mejores Hiperparámetros (XGBoost):")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)

if __name__ == "__main__":
    main()
