import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_data_clean.csv')
    val_path = os.path.join(base_dir, 'data', 'processed', 'val_data_clean.csv')
    
    print("Cargando los datos...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    target_col = 'price_by_night_person'
    
    # Se remueven columnas que son IDs, fechas o texto libre, y también variables de precio 
    # que representan "leakage" (información que no deberíamos saber o que es básicamente el target)
    cols_to_drop = [
        'searchid', 'date', 'name', 'detail', 'destination_name', 'destination_code', 
        'hid', 'geo_id', 'main_city_oid', 'date_search', 'date_ci',
        'price_by_night', 'price_by_night_adult', 'target'
    ]
    
    print("Separando variables...")
    X_train = df_train.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_train.columns], errors='ignore')
    y_train = df_train[target_col]
    
    X_val = df_val.drop(columns=[target_col] + [c for c in cols_to_drop if c in df_val.columns], errors='ignore')
    y_val = df_val[target_col]
    
    print("Limpieza: Imputando valores faltantes...")
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    # Variables continuas: imputar la media
    for col in num_cols:
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_val[col] = X_val[col].fillna(mean_val)
        
    # Variables categóricas: imputar la moda
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
    
    # Alinear columnas de train y val por si hay desajustes en las categorías presentes
    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    
    print("Entrenando Regresión Lineal (sin regularizar)...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("\n--- Importancia de Variables (Coeficientes Top 30) ---")
    coefs = pd.DataFrame({
        'Variable': X_train.columns,
        'Coeficiente': model.coef_,
        'Importancia_Absoluta': np.abs(model.coef_)
    }).sort_values(by='Importancia_Absoluta', ascending=False)
    
    print(coefs[['Variable', 'Coeficiente']].head(30))
    
    print("\nEvaluando en el set de validación...")
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Guardar modelo de forma nativa con joblib
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_model_path = os.path.join(models_dir, "linear_model_best.joblib")
    joblib.dump(model, out_model_path)
    print(f"Modelo seleccionado guardado en: {out_model_path}")

if __name__ == "__main__":
    main()
