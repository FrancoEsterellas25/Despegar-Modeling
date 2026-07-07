import pandas as pd
import numpy as np
import os
import sys
import json
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

def main():
    print("Iniciando Optimización Bayesiana para Modelos Tweedie (Local XGB, Local LGBM, Global LGBM)...")
    
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.VAL_DATA_PATH)
    
    target_col = 'price_by_night_person'
    cols_to_drop = [
        'searchid', 'date', 'name', 'detail', 'destination_name', 'destination_code', 
        'hid', 'geo_id', 'main_city_oid', 'date_search', 'date_ci',
        'price_by_night', 'price_by_night_adult', 'target', 'min_query_price', 'hotel_id_extra'
    ]
    
    X_train = df_train.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_train.columns], errors='ignore')
    y_train = df_train[target_col].values
    X_val = df_val.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_val.columns], errors='ignore')
    y_val = df_val[target_col].values
    
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
    
    cluster_feats = [c for c in X_train.columns if 'mca_dim' in c or 'dist_' in c or 'Rating' in c or c == 'numberOfRooms']
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train[cluster_feats].fillna(0))
    X_va_sc = scaler.transform(X_val[cluster_feats].fillna(0))
    
    dt_router = DecisionTreeRegressor(max_leaf_nodes=6, min_samples_leaf=1000, random_state=42)
    dt_router.fit(X_tr_sc, y_train)
    
    tr_labels = dt_router.apply(X_tr_sc)
    va_labels = dt_router.apply(X_va_sc)
    
    unique_leaves = np.unique(tr_labels)
    
    X_train['segmento'] = [f"Hoja_{i}" for i in tr_labels]
    X_train['TARGET'] = y_train
    X_val['segmento'] = [f"Hoja_{i}" for i in va_labels]
    X_val['TARGET'] = y_val
    
    segmentos = [f"Hoja_{i}" for i in unique_leaves]
    mejores_hiperparametros = {}
    
    # 1. TUNEAR MODELO GLOBAL LGBM
    print("\n--- TUNEANDO LIGHTGBM GLOBAL (TWEEDIE) ---")
    train_sub_sample, _ = train_test_split(X_train, train_size=0.1, random_state=42)
    val_sub_sample, _ = train_test_split(X_val, train_size=0.3, random_state=42)
    
    X_t_g = train_sub_sample.drop(columns=['TARGET', 'segmento']).values
    y_t_g = train_sub_sample['TARGET'].values
    X_v_g = val_sub_sample.drop(columns=['TARGET', 'segmento']).values
    y_v_g = val_sub_sample['TARGET'].values
    
    def objective_global(trial):
        params = {
            'objective': 'tweedie',
            'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'verbose': -1
        }
        lgbm = LGBMRegressor(**params, random_state=42, n_jobs=-1)
        lgbm.fit(X_t_g, y_t_g)
        preds = lgbm.predict(X_v_g)
        return r2_score(y_v_g, preds)
        
    study_global = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_global.optimize(objective_global, n_trials=30)
    mejores_hiperparametros["Global_LGBM"] = study_global.best_params
    
    # 2. TUNEAR EXPERTOS LOCALES (XGBoost y LGBM)
    for seg in segmentos:
        print(f"\n--- TUNEANDO EXPERTOS PARA {seg} ---")
        train_sub = X_train[X_train['segmento'] == seg]
        val_sub = X_val[X_val['segmento'] == seg]
        
        train_sub_sample, _ = train_test_split(train_sub, train_size=0.3, random_state=42)
        val_sub_sample = val_sub if len(val_sub) < 10000 else train_test_split(val_sub, train_size=10000/len(val_sub), random_state=42)[0]
        
        X_t = train_sub_sample.drop(columns=['TARGET', 'segmento']).values
        y_t = train_sub_sample['TARGET'].values
        X_v = val_sub_sample.drop(columns=['TARGET', 'segmento']).values
        y_v = val_sub_sample['TARGET'].values
        
        # XGBoost
        def objective_xgb(trial):
            params = {
                'objective': 'reg:tweedie',
                'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            xgb = XGBRegressor(**params, random_state=42, n_jobs=-1)
            xgb.fit(X_t, y_t)
            preds = xgb.predict(X_v)
            return r2_score(y_v, preds)
            
        study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study_xgb.optimize(objective_xgb, n_trials=30)
        mejores_hiperparametros[f"{seg}_XGB"] = study_xgb.best_params
        
        # LightGBM
        def objective_lgb(trial):
            params = {
                'objective': 'tweedie',
                'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'verbose': -1
            }
            lgb = LGBMRegressor(**params, random_state=42, n_jobs=-1)
            lgb.fit(X_t, y_t)
            preds = lgb.predict(X_v)
            return r2_score(y_v, preds)
            
        study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study_lgb.optimize(objective_lgb, n_trials=30)
        mejores_hiperparametros[f"{seg}_LGBM"] = study_lgb.best_params
        
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    out_path = os.path.join(config.MODELS_DIR, "best_params_tweedie.json")
    with open(out_path, "w") as f:
        json.dump(mejores_hiperparametros, f, indent=4)
        
    print(f"\n==================================================")
    print(f"¡OPTIMIZACIÓN FINALIZADA! Parámetros guardados en: {out_path}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
