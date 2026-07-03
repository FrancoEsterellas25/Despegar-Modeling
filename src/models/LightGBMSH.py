import pandas as pd
import numpy as np
import os
import random
import warnings
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# Ignorar advertencias inofensivas
warnings.filterwarnings("ignore", category=UserWarning)

def load_and_clean_data_lgbm(train_path, val_path):
    """
    Carga y limpieza adaptada para LightGBM.
    LightGBM también puede manejar categóricas de forma nativa si se les asigna
    el dtype 'category' de pandas. No necesita One-Hot Encoding.
    """
    print("Cargando los datos...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    print(f"Shape del DF de entrenamiento cargado: {df_train.shape}")
    
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
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Variables continuas: imputar la media
    for col in num_cols:
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_val[col] = X_val[col].fillna(mean_val)
        
    # Variables categóricas: imputar la moda y convertir a tipo 'category'
    # LightGBM reconoce automáticamente las columnas con dtype 'category'
    for col in cat_cols:
        if X_train[col].isnull().all():
            mode_val = "Missing"
        else:
            mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val).astype('category')
        X_val[col] = X_val[col].fillna(mode_val).astype('category')
    
    print(f"  Variables categóricas detectadas ({len(cat_cols)}): {cat_cols}")
    print("  -> LightGBM las detectará automáticamente por su dtype 'category'.")
    
    return X_train, y_train, X_val, y_val, cat_cols

def generate_random_configs_lgbm(n_configs=300):
    """
    Genera configuraciones aleatorias de hiperparámetros para LightGBM.
    Los pares (learning_rate, n_estimators) están atados para coherencia.
    """
    configs = []
    
    # Pares coherentes (learning_rate, n_estimators)
    lr_n_est_pairs = [
        (0.2, 100),   # Rápido, pocos árboles
        (0.1, 200),   # Balanceado estándar
        (0.08, 300),  # Ligeramente más lento
        (0.05, 500),  # Lento, moderados árboles
        (0.03, 700)   # Muy lento, bastantes árboles
    ]
    
    for _ in range(n_configs):
        seleccion = random.choice(lr_n_est_pairs)
        
        config = {
            'learning_rate': seleccion[0],
            'n_estimators': seleccion[1],
            'num_leaves': random.choice([15, 31, 63, 127, 255]),
            'max_depth': random.choice([-1, 5, 10, 15, 20]),  # -1 = sin límite
            'min_child_samples': random.choice([5, 10, 20, 50, 100]),
            'subsample': random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            'colsample_bytree': random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            'reg_alpha': random.choice([0, 0.1, 0.5, 1.0, 10.0]),
            'reg_lambda': random.choice([0, 0.1, 0.5, 1.0, 10.0]),
            'min_split_gain': random.choice([0.0, 0.01, 0.05, 0.1]),
        }
        configs.append(config)
    return configs

def evaluate_model_kfold_lgbm(config, X_subset, y_subset, cat_cols, n_splits=3):
    """
    Evalúa un modelo LightGBM usando K-Fold Cross Validation.
    Reconstruye DataFrames dentro de cada fold para preservar el dtype 'category'
    que LightGBM necesita para detectar las categóricas automáticamente.
    Retorna el RMSE promedio, MAE y R2 promedios.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, mae_scores, r2_scores = [], [], []
    
    col_names = X_subset.columns
    col_dtypes = X_subset.dtypes
    
    X_arr = X_subset.values
    y_arr = y_subset.values
    
    for train_idx, val_idx in kf.split(X_arr):
        # Reconstruir DataFrames para preservar dtypes categóricos
        X_kf_train = pd.DataFrame(X_arr[train_idx], columns=col_names)
        X_kf_val = pd.DataFrame(X_arr[val_idx], columns=col_names)
        
        # Restaurar dtypes originales (especialmente 'category')
        for col in col_names:
            X_kf_train[col] = X_kf_train[col].astype(col_dtypes[col])
            X_kf_val[col] = X_kf_val[col].astype(col_dtypes[col])
        
        y_kf_train, y_kf_val = y_arr[train_idx], y_arr[val_idx]
        
        model = lgb.LGBMRegressor(
            **config,
            random_state=42,
            n_jobs=-1,
            verbose=-1  # Silenciar salida de entrenamiento
        )
        model.fit(X_kf_train, y_kf_train)
        y_pred = model.predict(X_kf_val)
        
        rmse_scores.append(np.sqrt(mean_squared_error(y_kf_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_kf_val, y_pred))
        r2_scores.append(r2_score(y_kf_val, y_pred))
        
    return np.mean(rmse_scores), np.mean(mae_scores), np.mean(r2_scores)

def successive_halving_lgbm(X_train, y_train, cat_cols):
    print("\n" + "="*60)
    print("Iniciando algoritmo de Successive Halving (LightGBM)")
    print("="*60)
    
    configs = generate_random_configs_lgbm(300)
    current_resource = 4000
    eta = 2
    iteration = 0
    
    final_stats = {}
    
    # Loop hasta que queden exactamente 3 modelos.
    # 300 -> 150 -> 75 -> 37 -> 18 -> 9 -> 4 -> 3
    while len(configs) > 3:
        print(f"\n[Iteración {iteration}] Evaluando {len(configs)} modelos con recurso R = {current_resource} muestras.")
        
        if current_resource <= len(X_train):
            X_subset = X_train.sample(n=current_resource, random_state=iteration)
            y_subset = y_train.loc[X_subset.index]
        else:
            X_subset = X_train
            y_subset = y_train
            print("  -> Advertencia: El recurso superó el tamaño total de los datos. Usando todos los datos disponibles.")
        
        results = []
        for i, config in enumerate(configs):
            rmse, mae, r2 = evaluate_model_kfold_lgbm(config, X_subset, y_subset, cat_cols, n_splits=3)
            results.append({
                'config': config,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'eval_resource': current_resource
            })
            
            if (i + 1) % 10 == 0 or (i + 1) == len(configs):
                print(f"  Procesados {i+1}/{len(configs)} modelos...")
        
        # Ordenar por RMSE promedio (menor es mejor)
        results.sort(key=lambda x: x['rmse'])
        
        # Poda: quedarse con la mejor mitad (division entera por eta=2)
        n_survivors = len(configs) // eta
        if n_survivors < 3 and len(configs) > 3:
            n_survivors = 3 
            
        survivors = results[:n_survivors]
        configs = [s['config'] for s in survivors]
        
        print(f"  -> Peor RMSE podado: {results[-1]['rmse']:.4f}")
        print(f"  -> Mejor RMSE en esta iteración: {survivors[0]['rmse']:.4f}")
        print(f"  -> Modelos sobrevivientes: {len(configs)}")
        
        if len(configs) == 3:
            final_stats = survivors
        
        current_resource *= eta
        iteration += 1
        
    return configs, final_stats

def main():
    # Obtener el directorio raíz del proyecto (3 niveles arriba)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_data_clean.csv')
    val_path = os.path.join(base_dir, 'data', 'processed', 'val_data_clean.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    # 1. Carga y limpieza (adaptada para LightGBM, sin One-Hot Encoding)
    X_train, y_train, X_val, y_val, cat_cols = load_and_clean_data_lgbm(train_path, val_path)
    
    # 2. Ejecutar Successive Halving para LightGBM
    top_3_configs, top_3_stats = successive_halving_lgbm(X_train, y_train, cat_cols)
    
    # 3. Mostrar resumen de los 3 finalistas
    print("\n" + "*"*60)
    print("RESUMEN DE LOS 3 MODELOS FINALISTAS (última evaluación)")
    print("*"*60)
    for idx, stat in enumerate(top_3_stats):
        print(f"\nFinalista #{idx + 1}")
        print(f"  Hiperparámetros: {stat['config']}")
        print(f"  Métricas Cross-Validation (con R={stat['eval_resource']}) - RMSE: {stat['rmse']:.4f} | MAE: {stat['mae']:.4f} | R2: {stat['r2']:.4f}")
        
    # 4. Entrenar y probar los 3 finalistas sobre todo el set y predecir validación
    print("\n" + "*"*60)
    print("PREDICCIÓN FINAL EN EL SET DE VALIDACIÓN (val_data_clean.csv)")
    print("*"*60)
    
    model_metrics = []
    
    for idx, config in enumerate(top_3_configs):
        print(f"\nEntrenando Modelo Finalista #{idx + 1} con todo X_train...")
        model = lgb.LGBMRegressor(
            **config,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        print("Prediciendo en set de Validación...")
        y_pred = model.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"Métricas en val_data_clean.csv (Finalista #{idx + 1}):")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        
        stat_cv = top_3_stats[idx]
        model_metrics.append({
            'index': idx,
            'cv_rmse': stat_cv['rmse'],
            'cv_mae': stat_cv['mae'],
            'cv_r2': stat_cv['r2'],
            'val_rmse': rmse,
            'val_mae': mae,
            'model': model
        })
        
    # Sistema de votación de 5 métricas (mayoría simple)
    votes = [0, 0, 0]
    
    # 1. CV RMSE (Menor es mejor)
    votes[min(model_metrics, key=lambda x: x['cv_rmse'])['index']] += 1
    # 2. CV MAE (Menor es mejor)
    votes[min(model_metrics, key=lambda x: x['cv_mae'])['index']] += 1
    # 3. CV R2 (Mayor es mejor)
    votes[max(model_metrics, key=lambda x: x['cv_r2'])['index']] += 1
    # 4. Val RMSE (Menor es mejor)
    votes[min(model_metrics, key=lambda x: x['val_rmse'])['index']] += 1
    # 5. Val MAE (Menor es mejor)
    votes[min(model_metrics, key=lambda x: x['val_mae'])['index']] += 1
    
    best_model_idx = np.argmax(votes)
    print(f"\nResultados de la votación (Métricas ganadas por cada finalista):")
    for i in range(3):
        print(f"  * Finalista #{i+1}: {votes[i]} de 5 métricas.")
        
    best_model = model_metrics[best_model_idx]['model']
    print(f"\n>> El mejor modelo es el Finalista #{best_model_idx + 1} (ganador con {votes[best_model_idx]}/5 métricas).")
    
    # Guardar modelo de forma nativa
    os.makedirs(models_dir, exist_ok=True)
    out_model_path = os.path.join(models_dir, "lgb_model_best.txt")
    best_model.booster_.save_model(out_model_path)
    print(f"Modelo seleccionado guardado en: {out_model_path}")

if __name__ == "__main__":
    main()
