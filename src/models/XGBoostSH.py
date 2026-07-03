import pandas as pd
import numpy as np
import os
import random
import warnings
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# Ignorar advertencias inofensivas (si las hubiera)
warnings.filterwarnings("ignore", category=UserWarning)

def load_and_clean_data(train_path, val_path):
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
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    for col in num_cols:
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_val[col] = X_val[col].fillna(mean_val)
        
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

def generate_random_configs_xgb(n_configs=300):
    """Genera configuraciones aleatorias de hiperparámetros para XGBoost"""
    configs = []
    
    # Pares coherentes (learning_rate, n_estimators)
    # A menos árboles, mayor learning rate, y viceversa
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
            'max_depth': random.choice([3, 5, 7, 9, 12, 15]),
            'min_child_weight': random.choice([1, 3, 5, 7]),
            'subsample': random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            'colsample_bytree': random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            'reg_alpha': random.choice([0, 0.1, 0.5, 1.0, 10.0]),
            'reg_lambda': random.choice([0, 0.1, 0.5, 1.0, 10.0])
        }
        configs.append(config)
    return configs

def evaluate_model_kfold(model, X_subset, y_subset, n_splits=3):
    """
    Evalúa el modelo usando K-Fold Cross Validation.
    Retorna el RMSE promedio, y como métricas adicionales MAE y R2 promedios.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, mae_scores, r2_scores = [], [], []
    
    # Asegurar iteración sobre numpy arrays
    X_arr = X_subset.values
    y_arr = y_subset.values
    
    for train_idx, val_idx in kf.split(X_arr):
        X_kf_train, X_kf_val = X_arr[train_idx], X_arr[val_idx]
        y_kf_train, y_kf_val = y_arr[train_idx], y_arr[val_idx]
        
        # verbose=False para no ensuciar la salida por cada split
        model.fit(X_kf_train, y_kf_train, verbose=False)
        y_pred = model.predict(X_kf_val)
        
        rmse_scores.append(np.sqrt(mean_squared_error(y_kf_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_kf_val, y_pred))
        r2_scores.append(r2_score(y_kf_val, y_pred))
        
    return np.mean(rmse_scores), np.mean(mae_scores), np.mean(r2_scores)

def successive_halving_xgb(X_train, y_train):
    print("\n" + "="*60)
    print("Iniciando algoritmo de Successive Halving (XGBoost)")
    print("="*60)
    
    configs = generate_random_configs_xgb(300)
    current_resource = 3000
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
            # n_jobs=-1 para usar todos los procesadores
            model = XGBRegressor(**config, random_state=42, n_jobs=-1, eval_metric='rmse')
            rmse, mae, r2 = evaluate_model_kfold(model, X_subset, y_subset, n_splits=3)
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
    
    # 1. Carga y limpieza
    X_train, y_train, X_val, y_val = load_and_clean_data(train_path, val_path)
    
    # 2. Ejecutar Successive Halving para XGBoost
    top_3_configs, top_3_stats = successive_halving_xgb(X_train, y_train)
    
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
        model = XGBRegressor(**config, random_state=42, n_jobs=-1, eval_metric='rmse')
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
    out_model_path = os.path.join(models_dir, "xgb_model_best.json")
    best_model.save_model(out_model_path)
    print(f"Modelo seleccionado guardado en: {out_model_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
