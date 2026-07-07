import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, HuberRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

def main():
    print("======================================================")
    print(" INICIANDO PIPELINE DE STACKING (TWEEDIE + MAE)       ")
    print("======================================================")
    
    # ---------------------------------------------------------
    # PREPARACIÓN DE DATOS
    # ---------------------------------------------------------
    print("\nCargando datasets...")
    df_train1 = pd.read_csv(config.TRAIN_DATA_PATH)
    df_train2 = pd.read_csv(config.VAL_DATA_PATH)
    df_train = pd.concat([df_train1, df_train2], ignore_index=True)
    
    test_path = os.path.join(os.path.dirname(config.TRAIN_DATA_PATH), "test_data_clean.csv")
    df_test = pd.read_csv(test_path)
    
    target_col = 'price_by_night_person'
    cols_to_drop = [
        'searchid', 'date', 'name', 'detail', 'destination_name', 'destination_code', 
        'hid', 'geo_id', 'main_city_oid', 'date_search', 'date_ci',
        'price_by_night', 'price_by_night_adult', 'target', 'min_query_price', 'hotel_id_extra'
    ]
    
    # Sin filtrado de outliers (mantener todo el mundo real)
    
    X_train_full = df_train.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_train.columns], errors='ignore')
    y_train_full = df_train[target_col].values
    
    X_test_full = df_test.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_test.columns], errors='ignore')
    y_test_full = df_test[target_col].values
    
    print(f"Dimensiones de Entrenamiento Combinado: {X_train_full.shape}")
    print(f"Dimensiones de Test (Caja Fuerte): {X_test_full.shape}")
    
    # Imputación Simple
    print("Imputando valores faltantes...")
    num_cols = X_train_full.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns
    
    for col in num_cols:
        mean_val = X_train_full[col].mean()
        X_train_full[col] = X_train_full[col].fillna(mean_val)
        X_test_full[col] = X_test_full[col].fillna(mean_val)
        
    for col in cat_cols:
        mode_val = "Missing" if X_train_full[col].isnull().all() else X_train_full[col].mode()[0]
        X_train_full[col] = X_train_full[col].fillna(mode_val)
        X_test_full[col] = X_test_full[col].fillna(mode_val)
        
    X_train_full = pd.get_dummies(X_train_full, columns=cat_cols, drop_first=True)
    X_test_full = pd.get_dummies(X_test_full, columns=cat_cols, drop_first=True)
    
    for c in X_train_full.columns:
        if c not in X_test_full:
            X_test_full[c] = 0
    X_test_full = X_test_full[X_train_full.columns]
    
    # Parámetros Óptimos
    tuned_params = {}
    params_path = os.path.join(config.MODELS_DIR, "best_params_tweedie.json")
    if os.path.exists(params_path):
        print(f"Cargando hiperparámetros de Optuna desde {params_path}...")
        with open(params_path, "r") as f:
            tuned_params = json.load(f)
    else:
        print("ADVERTENCIA: No se encontraron hiperparámetros de Optuna pre-calculados.")
        
    # Variables de Enrutador
    cluster_feats = [c for c in X_train_full.columns if 'mca_dim' in c or 'dist_' in c or 'Rating' in c or c == 'numberOfRooms']
    
    # Matrices OOF
    OOF_XGBoost_Local = np.zeros(len(X_train_full))
    OOF_LightGBM_Local = np.zeros(len(X_train_full))
    OOF_LightGBM_Local_MAE = np.zeros(len(X_train_full))
    OOF_LightGBM_Global = np.zeros(len(X_train_full))
    OOF_Leaves = np.zeros(len(X_train_full))
    
    # ---------------------------------------------------------
    # FASE 1 A 3: BUCLE K-FOLD
    # ---------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nIniciando Bucle de Validación Cruzada (5-Fold)...")
    fold = 1
    
    for train_idx, val_idx in kf.split(X_train_full):
        print(f"\n====================== FOLD {fold}/5 ======================")
        X_tr = X_train_full.iloc[train_idx].copy()
        y_tr = y_train_full[train_idx]
        X_va = X_train_full.iloc[val_idx].copy()
        y_va = y_train_full[val_idx]
        
        # 1. Enrutador
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr[cluster_feats].fillna(0))
        X_va_sc = scaler.transform(X_va[cluster_feats].fillna(0))
        
        dt_router = DecisionTreeRegressor(max_leaf_nodes=6, min_samples_leaf=1000, random_state=42)
        dt_router.fit(X_tr_sc, y_tr)
        
        tr_leaves = dt_router.apply(X_tr_sc)
        va_leaves = dt_router.apply(X_va_sc)
        unique_leaves = np.unique(tr_leaves)
        
        X_tr['segmento'] = [f"Hoja_{i}" for i in tr_leaves]
        X_va['segmento'] = [f"Hoja_{i}" for i in va_leaves]
        OOF_Leaves[val_idx] = va_leaves
        
        # 2. Expertos Locales
        for leaf in unique_leaves:
            seg = f"Hoja_{leaf}"
            mask_tr = X_tr['segmento'] == seg
            mask_va = X_va['segmento'] == seg
            
            X_t_hoja = X_tr[mask_tr].drop(columns=['segmento']).values
            y_t_hoja = y_tr[mask_tr]
            
            X_v_hoja = X_va[mask_va].drop(columns=['segmento']).values
            y_v_hoja = y_va[mask_va]
            
            if len(X_t_hoja) == 0 or len(X_v_hoja) == 0:
                continue
                
            # XGBoost Local (Tweedie)
            p_xgb = tuned_params.get(f"{seg}_XGB", {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5, 'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5})
            xgb = XGBRegressor(**p_xgb, random_state=42, n_jobs=-1)
            xgb.fit(X_t_hoja, y_t_hoja, verbose=0)
            OOF_XGBoost_Local[val_idx[mask_va.values]] = xgb.predict(X_v_hoja)
            
            # LightGBM Local (Tweedie)
            p_lgb = tuned_params.get(f"{seg}_LGBM", {'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5, 'verbose': -1})
            p_lgb['verbose'] = -1
            lgb = LGBMRegressor(**p_lgb, random_state=42, n_jobs=-1)
            lgb.fit(X_t_hoja, y_t_hoja)
            OOF_LightGBM_Local[val_idx[mask_va.values]] = lgb.predict(X_v_hoja)
            
            # LightGBM Local (MAE)
            p_lgb_mae = p_lgb.copy()
            p_lgb_mae['objective'] = 'mae'
            p_lgb_mae.pop('tweedie_variance_power', None)
            lgb_mae = LGBMRegressor(**p_lgb_mae, random_state=42, n_jobs=-1)
            lgb_mae.fit(X_t_hoja, y_t_hoja)
            OOF_LightGBM_Local_MAE[val_idx[mask_va.values]] = lgb_mae.predict(X_v_hoja)
            
        # 3. LightGBM Global
        print(f"-> Entrenando LightGBM Global (Fold {fold})...")
        X_t_g = X_tr.drop(columns=['segmento']).values
        X_v_g = X_va.drop(columns=['segmento']).values
        
        p_glob = tuned_params.get("Global_LGBM", {'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5, 'verbose': -1})
        p_glob['verbose'] = -1
        lgb_global = LGBMRegressor(**p_glob, random_state=42, n_jobs=-1)
        lgb_global.fit(X_t_g, y_tr)
        OOF_LightGBM_Global[val_idx] = lgb_global.predict(X_v_g)
        
        fold += 1

    # ---------------------------------------------------------
    # FASE 4: STACKING (META-MODELOS MÚLTIPLES)
    # ---------------------------------------------------------
    print("\n======================================================")
    print(" ENTRENANDO META-MODELOS (EL CUADRANTE MÁGICO)        ")
    print("======================================================")
    
    # Matriz 1: Sin Especialista MAE (Solo Tweedie)
    X_meta_train_puro = np.column_stack((
        OOF_XGBoost_Local,
        OOF_LightGBM_Local,
        OOF_LightGBM_Global
    ))
    
    # Matriz 2: Con Especialista MAE
    X_meta_train_hibrido = np.column_stack((
        OOF_XGBoost_Local,
        OOF_LightGBM_Local,
        OOF_LightGBM_Local_MAE,
        OOF_LightGBM_Global
    ))
    
    # Matriz 3: Regresión Lineal Condicional (Interacciones)
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    OOF_Leaves_OHE = ohe.fit_transform(OOF_Leaves.reshape(-1, 1))
    
    # Crear interacciones: multiplicar cada predicción por cada hoja One-Hot
    X_interacciones_train = []
    for pred_col in [OOF_XGBoost_Local, OOF_LightGBM_Local, OOF_LightGBM_Local_MAE, OOF_LightGBM_Global]:
        # pred_col tiene shape (N,)
        # Multiplicamos la columna por toda la matriz OHE de forma broadcasteada
        interaccion = OOF_Leaves_OHE * pred_col.reshape(-1, 1)
        X_interacciones_train.append(interaccion)
        
    X_meta_train_lineal_condicional = np.column_stack(
        [X_meta_train_hibrido] + X_interacciones_train
    )
    
    # Definir Jueces
    ridge_puro = Ridge(alpha=1.0).fit(X_meta_train_puro, y_train_full)
    huber_puro = HuberRegressor(max_iter=1000).fit(X_meta_train_puro, y_train_full)
    
    ridge_hibrido = Ridge(alpha=1.0).fit(X_meta_train_hibrido, y_train_full)
    huber_hibrido = HuberRegressor(max_iter=1000).fit(X_meta_train_hibrido, y_train_full)
    
    # Meta-Modelo Lineal Condicional (Ridge con regularización fuerte)
    ridge_condicional = Ridge(alpha=10.0).fit(X_meta_train_lineal_condicional, y_train_full)
    
    # ---------------------------------------------------------
    # FASE 5: INFERENCIA SOBRE TEST (CAJA FUERTE)
    # ---------------------------------------------------------
    print("\n======================================================")
    print(" ENTRENANDO MODELOS FINALES EN EL 100% Y EVALUANDO TEST ")
    print("======================================================")
    
    # 1. Enrutador Final
    scaler_final = StandardScaler()
    X_train_sc_final = scaler_final.fit_transform(X_train_full[cluster_feats].fillna(0))
    X_test_sc_final = scaler_final.transform(X_test_full[cluster_feats].fillna(0))
    
    dt_router_final = DecisionTreeRegressor(max_leaf_nodes=6, min_samples_leaf=1000, random_state=42)
    dt_router_final.fit(X_train_sc_final, y_train_full)
    
    tr_leaves_final = dt_router_final.apply(X_train_sc_final)
    te_leaves_final = dt_router_final.apply(X_test_sc_final)
    
    unique_leaves_final = np.unique(tr_leaves_final)
    
    X_train_full['segmento'] = [f"Hoja_{i}" for i in tr_leaves_final]
    X_test_full['segmento'] = [f"Hoja_{i}" for i in te_leaves_final]
    
    Preds_XGB_Test = np.zeros(len(X_test_full))
    Preds_LGB_Test = np.zeros(len(X_test_full))
    Preds_LGB_MAE_Test = np.zeros(len(X_test_full))
    
    # 2. Expertos Finales
    for leaf in unique_leaves_final:
        seg = f"Hoja_{leaf}"
        mask_tr = X_train_full['segmento'] == seg
        mask_te = X_test_full['segmento'] == seg
        
        X_t_hoja = X_train_full[mask_tr].drop(columns=['segmento']).values
        y_t_hoja = y_train_full[mask_tr]
        X_te_hoja = X_test_full[mask_te].drop(columns=['segmento']).values
        
        if len(X_t_hoja) == 0:
            continue
            
        # Final XGB
        p_xgb = tuned_params.get(f"{seg}_XGB", {'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5, 'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5})
        xgb_f = XGBRegressor(**p_xgb, random_state=42, n_jobs=-1)
        xgb_f.fit(X_t_hoja, y_t_hoja, verbose=0)
        
        # Final LGBM Tweedie
        p_lgb = tuned_params.get(f"{seg}_LGBM", {'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5, 'verbose': -1})
        p_lgb['verbose'] = -1
        lgb_f = LGBMRegressor(**p_lgb, random_state=42, n_jobs=-1)
        lgb_f.fit(X_t_hoja, y_t_hoja)
        
        # Final LGBM MAE
        p_lgb_mae = p_lgb.copy()
        p_lgb_mae['objective'] = 'mae'
        p_lgb_mae.pop('tweedie_variance_power', None)
        lgb_mae_f = LGBMRegressor(**p_lgb_mae, random_state=42, n_jobs=-1)
        lgb_mae_f.fit(X_t_hoja, y_t_hoja)
        
        if len(X_te_hoja) > 0:
            Preds_XGB_Test[mask_te.values] = xgb_f.predict(X_te_hoja)
            Preds_LGB_Test[mask_te.values] = lgb_f.predict(X_te_hoja)
            Preds_LGB_MAE_Test[mask_te.values] = lgb_mae_f.predict(X_te_hoja)
            
    # 3. Global Final
    X_t_g_final = X_train_full.drop(columns=['segmento']).values
    X_te_g_final = X_test_full.drop(columns=['segmento']).values
    
    p_glob = tuned_params.get("Global_LGBM", {'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5, 'verbose': -1})
    p_glob['verbose'] = -1
    lgb_g_final = LGBMRegressor(**p_glob, random_state=42, n_jobs=-1)
    lgb_g_final.fit(X_t_g_final, y_train_full)
    
    Preds_Global_Test = lgb_g_final.predict(X_te_g_final)
    
    # 4. Matrices Finales
    X_meta_test_puro = np.column_stack((
        Preds_XGB_Test,
        Preds_LGB_Test,
        Preds_Global_Test
    ))
    
    X_meta_test_hibrido = np.column_stack((
        Preds_XGB_Test,
        Preds_LGB_Test,
        Preds_LGB_MAE_Test,
        Preds_Global_Test
    ))
    
    # Matriz 3 Test: Interacciones
    te_leaves_final_ohe = ohe.transform(te_leaves_final.reshape(-1, 1))
    X_interacciones_test = []
    for pred_col in [Preds_XGB_Test, Preds_LGB_Test, Preds_LGB_MAE_Test, Preds_Global_Test]:
        interaccion_test = te_leaves_final_ohe * pred_col.reshape(-1, 1)
        X_interacciones_test.append(interaccion_test)
        
    X_meta_test_lineal_condicional = np.column_stack(
        [X_meta_test_hibrido] + X_interacciones_test
    )
    
    # 5. Generar Reporte de Cuadrante
    res_ridge_puro = ridge_puro.predict(X_meta_test_puro)
    res_huber_puro = huber_puro.predict(X_meta_test_puro)
    res_ridge_hibrido = ridge_hibrido.predict(X_meta_test_hibrido)
    res_huber_hibrido = huber_hibrido.predict(X_meta_test_hibrido)
    res_ridge_condicional = ridge_condicional.predict(X_meta_test_lineal_condicional)
    
    print("\n" + "*"*60)
    print(" EL CUADRANTE MÁGICO Y META-MODELO LINEAL CONDICIONAL (TEST)")
    print("*"*60)
    print(f"[1] Tweedie Puro + RIDGE:             R2={r2_score(y_test_full, res_ridge_puro):.4f} | MAE={mean_absolute_error(y_test_full, res_ridge_puro):.4f}")
    print(f"[2] Tweedie Puro + HUBER:             R2={r2_score(y_test_full, res_huber_puro):.4f} | MAE={mean_absolute_error(y_test_full, res_huber_puro):.4f}")
    print(f"[3] Tweedie+MAE  + RIDGE:             R2={r2_score(y_test_full, res_ridge_hibrido):.4f} | MAE={mean_absolute_error(y_test_full, res_ridge_hibrido):.4f}")
    print(f"[4] Tweedie+MAE  + HUBER:             R2={r2_score(y_test_full, res_huber_hibrido):.4f} | MAE={mean_absolute_error(y_test_full, res_huber_hibrido):.4f}")
    print(f"[5] Tweedie+MAE  + LINEAL CONDICIONAL:R2={r2_score(y_test_full, res_ridge_condicional):.4f} | MAE={mean_absolute_error(y_test_full, res_ridge_condicional):.4f}")
    print("*"*60)
    
    # Exportar predicciones para la Interfaz de Streamlit
    os.makedirs("data/cache", exist_ok=True)
    
    hotel_names = df_test['name'] if 'name' in df_test.columns else [f"Hotel {i}" for i in range(len(df_test))]
    df_test_export = pd.DataFrame({
        'hotel': hotel_names,
        'precio_real': y_test_full,
        'prediccion_huber': res_huber_hibrido,
        'prediccion_ridge': res_ridge_puro
    })
    df_test_export.to_csv('data/cache/test_predictions.csv', index=False)
    print("\nPredicciones exportadas a data/cache/test_predictions.csv para auditoría en UI.")

if __name__ == "__main__":
    main()
