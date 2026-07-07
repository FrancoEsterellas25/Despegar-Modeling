import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import config

def main():
    print("="*60)
    print(" BASELINE: REGRESIÓN LINEAL (RIDGE) GLOBAL ")
    print("="*60)
    
    # 1. Cargar Datos
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
    
    y_train = df_train[target_col].values
    X_train = df_train.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_train.columns], errors='ignore')
    
    y_test = df_test[target_col].values
    X_test = df_test.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_test.columns], errors='ignore')
    
    print(f"Dimensiones de Entrenamiento: {X_train.shape}")
    print(f"Dimensiones de Test: {X_test.shape}")
    
    # 2. Pipeline simple: Imputar nulos con mediana, escalar y correr Ridge
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ])
    
    print("\nEntrenando Baseline Lineal...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluando en Test...")
    preds = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print("\n" + "*"*60)
    print(" RESULTADOS DEL BASELINE LINEAL (100% de los datos)")
    print("*"*60)
    print(f"R2:  {r2:.4f}")
    print(f"MAE: ${mae:.4f} USD")
    print("*"*60)

if __name__ == "__main__":
    main()
