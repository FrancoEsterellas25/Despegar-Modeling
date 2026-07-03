import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# Ignorar advertencias inofensivas
warnings.filterwarnings("ignore", category=UserWarning)

def load_and_clean_data(train_path, val_path):
    print("Cargando los datos...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    print(f"Shape del DF de entrenamiento: {df_train.shape}")
    print(f"Shape del DF de validación: {df_val.shape}")
    
    target_col = 'price_by_night_person'
    cols_to_drop = [
        'searchid', 'date', 'name', 'detail', 'destination_name', 'destination_code', 
        'hid', 'geo_id', 'main_city_oid', 'date_search', 'date_ci',
        'price_by_night', 'price_by_night_adult', 'target', 'min_query_price'
    ]
    
    print("Separando variables y eliminando columnas de 'leakage'...")
    X_train = df_train.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_train.columns], errors='ignore')
    y_train = df_train[target_col]
    
    X_val = df_val.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_val.columns], errors='ignore')
    y_val = df_val[target_col]
    
    print("Limpieza: Imputando valores faltantes...")
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    # Imputación de variables numéricas por media
    for col in num_cols:
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_val[col] = X_val[col].fillna(mean_val)
        
    # Imputación de variables categóricas por moda
    for col in cat_cols:
        if X_train[col].isnull().all():
            mode_val = "Missing"
        else:
            mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_val[col] = X_val[col].fillna(mode_val)
        
    print("Codificando variables categóricas (One-Hot Encoding)...")
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
    
    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    
    return X_train, y_train, X_val, y_val

def get_base_models():
    """
    Define los modelos del Nivel 0 con las configuraciones optimizadas por el usuario.
    """
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=7,
        min_samples_leaf=2,
        max_features=0.4,
        random_state=42,
        n_jobs=-1
    )
    
    xgb = XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=9,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'
    )
    
    lgbm = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=10,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_split_gain=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    return {
        'RandomForest': rf,
        'XGBoost': xgb,
        'LightGBM': lgbm
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'train_data_clean.csv')
    val_path = os.path.join(base_dir, 'val_data_clean.csv')
    
    # 1. Cargar y limpiar datos
    X_train, y_train, X_val, y_val = load_and_clean_data(train_path, val_path)
    
    # Para agilizar el cálculo en el script (K=5 folds sobre todo el dataset de 1M+ de filas con RF/XGB/LGB puede tomar horas)
    # Hacemos un submuestreo de 100,000 registros para train para que sea rápido y ejecutable en tiempo prudencial.
    max_train_samples = 800000
    if len(X_train) > max_train_samples:
        print(f"\nSubmuestreando X_train a {max_train_samples} filas para hacer viable la ejecución del Stacking...")
        X_train_sub = X_train.sample(n=max_train_samples, random_state=42)
        y_train_sub = y_train.loc[X_train_sub.index]
    else:
        X_train_sub = X_train
        y_train_sub = y_train
        
    X_train_arr = X_train_sub.values
    y_train_arr = y_train_sub.values
    
    # 2. Configurar Cross Validation
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    # 3. Obtener Modelos Base
    base_models = get_base_models()
    
    # Matrices para guardar predicciones Out-Of-Fold (OOF) del Nivel 1
    # Estas matrices tendrán forma (n_muestras, n_modelos_base)
    oof_train = np.zeros((len(X_train_sub), len(base_models)))
    oof_val = np.zeros((len(X_val), len(base_models)))
    
    print("\n" + "="*60)
    print("NIVEL 0: Entrenando modelos base y generando predicciones OOF (K=5)...")
    print("="*60)
    
    for model_idx, (name, model) in enumerate(base_models.items()):
        print(f"\nProcesando modelo base: {name}...")
        
        # Array temporal para guardar las predicciones del fold de validación
        oof_predictions = np.zeros(len(X_train_sub))
        # Lista para acumular las predicciones del set de validación final en cada fold (para promediar)
        val_preds_fold = np.zeros(len(X_val))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_arr)):
            X_tr, y_tr = X_train_arr[train_idx], y_train_arr[train_idx]
            X_va = X_train_arr[val_idx]
            
            # Entrenar en K-1 folds
            model.fit(X_tr, y_tr)
            
            # Predecir en el fold restante
            oof_predictions[val_idx] = model.predict(X_va)
            
            # Predecir en el set de validación final externa
            val_preds_fold += model.predict(X_val.values) / K
            
            print(f"  Fold {fold+1}/{K} completado.")
            
        # Almacenar en las matrices OOF
        oof_train[:, model_idx] = oof_predictions
        oof_val[:, model_idx] = val_preds_fold
        
        # Calcular performance individual OOF
        rmse_oof = np.sqrt(mean_squared_error(y_train_arr, oof_predictions))
        mae_oof = mean_absolute_error(y_train_arr, oof_predictions)
        print(f">> {name} OOF Performance: RMSE={rmse_oof:.4f} | MAE={mae_oof:.4f}")

    # Convertir OOF a DataFrame para comodidad
    oof_train_df = pd.DataFrame(oof_train, columns=list(base_models.keys()))
    oof_val_df = pd.DataFrame(oof_val, columns=list(base_models.keys()))
    
    print("\n" + "="*60)
    print("CORRELACIÓN DE PREDICCIONES OUT-OF-FOLD (OOF) ENTRE MODELOS BASE")
    print("="*60)
    correlation_matrix = oof_train_df.corr()
    print(correlation_matrix)
    
    print("\n" + "="*60)
    print("NIVEL 1: Entrenando el meta-modelo...")
    print("="*60)
    
    # ELECCIÓN DEL META-MODELO:
    # Usamos una Regresión Lineal simple (OLS) sin regularización a petición del usuario.
    # Nota: la correlación entre las predicciones de los modelos base suele ser muy alta, 
    # por lo que los coeficientes resultantes mostrarán la importancia directa que el
    # modelo de Stacking asigna a cada predictor, aunque hay que vigilar el potencial 
    # impacto de la multicolinealidad en la estabilidad de estos coeficientes.
    
    print("Entrenando Regresión Lineal ordinaria (sin regularización) en OOF de Nivel 0...")
    meta_model = LinearRegression()
    meta_model.fit(oof_train_df, y_train_arr)
    
    print("  Pesos (Importancia de Variables) asignados a cada modelo base:")
    for name, weight in zip(base_models.keys(), meta_model.coef_):
        print(f"    - {name}: {weight:.4f}")
    print(f"    - Intercepto: {meta_model.intercept_:.4f}")
    
    # 4. Evaluación en el set de validación final (val_data_clean.csv)
    print("\n" + "="*60)
    print("EVALUACIÓN DE RENDIMIENTO EN EL SET DE VALIDACIÓN EXTERNO")
    print("="*60)
    
    # Evaluar modelos base en validación final externa
    best_base_rmse = float('inf')
    best_base_name = ""
    
    for name_idx, name in enumerate(base_models.keys()):
        preds = oof_val_df[name]
        rmse_val = np.sqrt(mean_squared_error(y_val, preds))
        mae_val = mean_absolute_error(y_val, preds)
        r2_val = r2_score(y_val, preds)
        print(f"Modelo Base {name} en Validación:")
        print(f"  MAE: {mae_val:.4f} | RMSE: {rmse_val:.4f} | R2: {r2_val:.4f}")
        if rmse_val < best_base_rmse:
            best_base_rmse = rmse_val
            best_base_name = name
            
    # Evaluar Stacking final
    stack_preds = meta_model.predict(oof_val_df)
    rmse_stack = np.sqrt(mean_squared_error(y_val, stack_preds))
    mae_stack = mean_absolute_error(y_val, stack_preds)
    r2_stack = r2_score(y_val, stack_preds)
    
    print(f"\nModelo Stacking (Meta-modelo: Regresión Lineal ordinaria) en Validación:")
    print(f"  MAE: {mae_stack:.4f} | RMSE: {rmse_stack:.4f} | R2: {r2_stack:.4f}")
    
    improvement = best_base_rmse - rmse_stack
    print(f"\nAnálisis de Mejora:")
    if improvement > 0:
        print(f"  ¡Éxito! El modelo Stacking superó al mejor modelo individual ({best_base_name}) por {improvement:.4f} puntos de RMSE.")
    else:
        print(f"  El Stacking no superó al mejor modelo individual ({best_base_name}). Diferencia RMSE: {improvement:.4f}")

    # Serializar el modelo de Stacking (Meta-modelo y Modelos Base entrenados)
    print("\n" + "="*60)
    print("GUARDANDO MODELO DE STACKING...")
    print("="*60)
    
    # Entrenar modelos base en todo el dataset de entrenamiento submuestreado
    fitted_base_models = {}
    for name, model in base_models.items():
        print(f"Entrenando {name} sobre todo el dataset de entrenamiento...")
        model.fit(X_train_arr, y_train_arr)
        fitted_base_models[name] = model
        
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_model_path = os.path.join(models_dir, "stacking_model_best.joblib")
    
    stacking_data = {
        'meta_model': meta_model,
        'base_models': fitted_base_models
    }
    
    joblib.dump(stacking_data, out_model_path)
    print(f"Modelo Stacking guardado exitosamente en: {out_model_path}")

if __name__ == "__main__":
    main()
