import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # Cargar los datasets
    # El dataset base es 'train_data_clean.csv' y el aumentado es 'train_data_symbolic.csv'
    train_base_path = os.path.join(processed_dir, "train_data_clean.csv")
    train_sym_path = os.path.join(processed_dir, "train_data_symbolic.csv")
    
    if not os.path.exists(train_sym_path):
        raise FileNotFoundError(
            f"No se encontró el dataset aumentado en {train_sym_path}. "
            "Por favor, ejecuta primero: py src/data/generate_symbolic_features.py"
        )
        
    print("Cargando datasets para el test A/B de variables simbólicas...")
    df_base = pd.read_csv(train_base_path)
    df_sym = pd.read_csv(train_sym_path)
    
    # Definir variables de entrada comunes
    features_base = [
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
        features_base.append(f"mca_dim_{i}")
        
    # Variables del dataset aumentado (originales + 3 nuevas features simbólicas)
    features_sym = features_base + ["symbolic_feat_1", "symbolic_feat_2", "symbolic_feat_3"]
    
    target_col = "price_by_night_person"
    
    # Reducimos el tamaño para la validación cruzada para que corra rápidamente
    sample_size = 100000
    if len(df_base) > sample_size:
        print(f"Tomando una muestra aleatoria de {sample_size} registros para validación cruzada rápida...")
        sample_indices = df_base.sample(n=sample_size, random_state=42).index
        df_base_sub = df_base.loc[sample_indices].reset_index(drop=True)
        df_sym_sub = df_sym.loc[sample_indices].reset_index(drop=True)
    else:
        df_base_sub = df_base
        df_sym_sub = df_sym
        
    X_base = df_base_sub[features_base]
    X_sym = df_sym_sub[features_sym]
    y = df_base_sub[target_col]
    
    # Configuración estricta de validación
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    # XGBoost Baseline restrictivo (evita interacciones complejas por su cuenta)
    xgb_base = XGBRegressor(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse"
    )
    
    # Listas para almacenar métricas por fold
    rmse_base_folds = []
    rmse_sym_folds = []
    
    print("\n============================================================")
    print("PROCESANDO TEST A/B DE COMPROBACIÓN EMPÍRICA (K=5)...")
    print("============================================================")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_base)):
        print(f"\nProcesando Fold {fold + 1}/{K}...")
        
        # 1. Baseline: Dataset Original
        X_tr_b, y_tr_b = X_base.iloc[train_idx], y.iloc[train_idx]
        X_va_b, y_va_b = X_base.iloc[val_idx], y.iloc[val_idx]
        
        xgb_base.fit(X_tr_b, y_tr_b)
        pred_b = xgb_base.predict(X_va_b)
        rmse_b = np.sqrt(mean_squared_error(y_va_b, pred_b))
        rmse_base_folds.append(rmse_b)
        
        # 2. Modelo Aumentado: Dataset con Variables Simbólicas
        X_tr_s, y_tr_s = X_sym.iloc[train_idx], y.iloc[train_idx]
        X_va_s, y_va_s = X_sym.iloc[val_idx], y.iloc[val_idx]
        
        xgb_base.fit(X_tr_s, y_tr_s)
        pred_s = xgb_base.predict(X_va_s)
        rmse_s = np.sqrt(mean_squared_error(y_va_s, pred_s))
        rmse_sym_folds.append(rmse_s)
        
        print(f"  * RMSE Dataset Original:  {rmse_b:.4f}")
        print(f"  * RMSE Dataset Aumentado: {rmse_s:.4f}")
        
    mean_rmse_base = np.mean(rmse_base_folds)
    mean_rmse_sym = np.mean(rmse_sym_folds)
    std_rmse_base = np.std(rmse_base_folds)
    std_rmse_sym = np.std(rmse_sym_folds)
    
    print("\n============================================================")
    print("RESULTADOS GLOBALES DE RENDIMIENTO (CV PROMEDIO)")
    print("============================================================")
    print(f"  * Dataset Original:  {mean_rmse_base:.4f} (± {std_rmse_base:.4f})")
    print(f"  * Dataset Aumentado: {mean_rmse_sym:.4f} (± {std_rmse_sym:.4f})")
    
    # PRUEBA DE HIPÓTESIS ESTADÍSTICA: Paired T-Test
    # Comprobamos si la diferencia de error en los mismos folds es estadísticamente significativa
    # (p-valor < 0.05 rechaza la hipótesis nula de que ambos datasets se comportan igual).
    t_stat, p_val = ttest_rel(rmse_base_folds, rmse_sym_folds)
    
    print(f"  * Diferencia Absoluta de RMSE: {mean_rmse_base - mean_rmse_sym:.4f}")
    print(f"  * T-Statistic (t-test pareado): {t_stat:.4f}")
    print(f"  * P-Valor resultante:            {p_val:.6f}")
    
    if p_val < 0.05 and mean_rmse_sym < mean_rmse_base:
        print("\n>> VERDICTO: ¡La mejora es ESTADÍSTICAMENTE SIGNIFICATIVA! (p-valor < 0.05).")
        print("   La incorporación de las variables simbólicas aporta ganancia de información real.")
    else:
        print("\n>> VERDICTO: La diferencia NO es estadísticamente significativa a nivel de confianza de 95%.")
        print("   Podría ser ruido debido a la varianza de los folds.")
        
    # 5. AUTOPSIA: Entrenar en todo el dataset de muestra y ver Feature Importances
    print("\n============================================================")
    print("AUTOPSIA DE CARACTERÍSTICAS (IMPORTANCIA DE VARIABLES)")
    print("============================================================")
    print("Entrenando modelo final en todo el dataset aumentado...")
    xgb_base.fit(X_sym, y)
    
    importances = xgb_base.feature_importances_
    df_importances = pd.DataFrame({
        "Feature": features_sym,
        "Importancia": importances
    }).sort_values(by="Importancia", ascending=False).head(5)
    
    print("Top 5 Variables más importantes elegidas por XGBoost:")
    for idx, row in df_importances.iterrows():
        print(f"  {row['Feature']}: {row['Importancia']:.4f}")

if __name__ == "__main__":
    main()
