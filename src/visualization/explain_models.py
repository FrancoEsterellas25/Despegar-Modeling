import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Aplicar estilo profesional y limpio
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def plot_predicted_vs_actual(y_true, y_preds_dict, save_path=None):
    """
    Gráfico 1: Predicho vs. Real (Diagnóstico de Residuos) en grilla 2x2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.ravel()
    
    # Asegurar que los datos sean arrays
    y_true_arr = np.array(y_true)
    
    for idx, (model_name, y_pred) in enumerate(y_preds_dict.items()):
        ax = axes[idx]
        y_pred_arr = np.array(y_pred)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
        r2 = r2_score(y_true_arr, y_pred_arr)
        
        # Scatter plot con transparencia para visualizar densidad geométrica
        ax.scatter(y_true_arr, y_pred_arr, alpha=0.3, color='#2c3e50', edgecolors='none', s=25)
        
        # Línea de referencia y = x (predicción perfecta)
        min_val = min(y_true_arr.min(), y_pred_arr.min())
        max_val = max(y_true_arr.max(), y_pred_arr.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2.5, label='Predicción Perfecta')
        
        # Configurar límites y etiquetas
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_title(f"Modelo: {model_name}", fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel("Valor Real (y_test)", fontsize=13)
        ax.set_ylabel("Valor Predicho (y_pred)", fontsize=13)
        
        # Incrustar métricas en una esquina
        textstr = '\n'.join((
            f'RMSE: {rmse:.4f}',
            f'$R^2$: {r2:.4f}'
        ))
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, fontweight='bold')
        
        ax.legend(loc='lower right', fontsize=11)
        ax.tick_params(labelsize=11)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico Predicho vs Real guardado en: {save_path}")
    plt.close()

def plot_feature_importances(base_models_dict, meta_model, X_train, X_test, feature_names, save_dir=None):
    """
    Gráfico 2: Explicabilidad e Importancia de variables.
    - Utiliza SHAP para los modelos base (RF, XGB, LGBM).
    - Grafica coeficientes directos del Meta-modelo (Ridge).
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    # 1. Calcular y graficar importancia SHAP para los 3 modelos base
    for name, model in base_models_dict.items():
        print(f"\nCalculando SHAP TreeExplainer para: {name}...")
        try:
            # SHAP explainer optimizado para modelos basados en árboles (TreeExplainer)
            explainer = shap.TreeExplainer(model)
            # Muestreamos X_test para acelerar el cálculo si es muy grande
            sample_size = min(500, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42) if hasattr(X_test, 'sample') else X_test[:sample_size]
            
            shap_values = explainer(X_sample)
            
            plt.figure(figsize=(12, 8))
            plt.title(f"Top 5 Variables Importantes (SHAP) - {name}", fontsize=16, fontweight='bold', pad=15)
            # Mostrar solo el top 5
            shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=5, show=False)
            
            # Ajustar etiquetas de fuentes
            plt.xlabel("Valor Medio Absoluto SHAP (Impacto en la predicción)", fontsize=13)
            plt.tick_params(labelsize=12)
            
            if save_dir:
                path = os.path.join(save_dir, f"shap_importance_{name.lower()}.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"  -> SHAP Plot guardado en: {path}")
            plt.close()
            
        except Exception as e:
            print(f"No se pudo calcular SHAP para {name}: {e}")
            
    # 2. Graficar coeficientes del Meta-modelo (Ridge)
    print("\nGraficando coeficientes del Meta-modelo Ridge...")
    weights = meta_model.coef_
    base_names = list(base_models_dict.keys())
    
    plt.figure(figsize=(10, 6))
    # Paleta de colores profesionales
    colors = ['#34495e', '#3498db', '#2ecc71']
    
    sns.barplot(x=base_names, y=weights, palette=colors, edgecolor='#2c3e50', linewidth=1.5)
    
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.title("Pesos Asignados por el Meta-modelo (Ridge)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Coeficiente (Peso relativo)", fontsize=13)
    plt.xlabel("Modelos Base (Nivel 0)", fontsize=13)
    plt.tick_params(labelsize=12)
    
    # Agregar valores encima de las barras
    for idx, w in enumerate(weights):
        plt.text(idx, w + (0.01 if w >= 0 else -0.04), f"{w:.4f}", ha='center', fontweight='bold', fontsize=12)
        
    if save_dir:
        path = os.path.join(save_dir, "meta_model_coefficients.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  -> Coeficientes de Ridge guardados en: {path}")
    plt.close()

def main():
    # Obtener el directorio raíz
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    figures_dir = os.path.join(base_dir, "reports", "figures")
    
    # Cargar datos
    train_path = os.path.join(processed_dir, "train_data_clean.csv")
    val_path = os.path.join(processed_dir, "val_data_clean.csv")
    
    print("Cargando datasets para entrenamiento y evaluación visual...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Definir variables de entrada
    features = [
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
        features.append(f"mca_dim_{i}")
        
    target_col = "price_by_night_person"
    
    # Submuestrear para optimizar tiempos de entrenamiento y visualización
    sample_train = 50000
    sample_val = 10000
    
    X_train = train_df[features].sample(n=min(sample_train, len(train_df)), random_state=42)
    y_train = train_df.loc[X_train.index, target_col]
    
    X_val = val_df[features].sample(n=min(sample_val, len(val_df)), random_state=42)
    y_val = val_df.loc[X_val.index, target_col]
    
    # Hiperparámetros óptimos de los modelos base
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='rmse')
    lgbm = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    
    base_models = {
        'RandomForest': rf,
        'XGBoost': xgb,
        'LightGBM': lgbm
    }
    
    print("\nEntrenando nivel 0 y generando predicciones OOF (K=5)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Matrices OOF
    oof_train = np.zeros((len(X_train), len(base_models)))
    oof_val = np.zeros((len(X_val), len(base_models)))
    
    # Entrenar base models y estimar predicciones
    for model_idx, (name, model) in enumerate(base_models.items()):
        print(f"  * Entrenando {name}...")
        val_preds_fold = np.zeros(len(X_val))
        
        for train_idx, val_idx_kf in kf.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va_kf = X_train.iloc[val_idx_kf]
            
            model.fit(X_tr, y_tr)
            oof_train[val_idx_kf, model_idx] = model.predict(X_va_kf)
            val_preds_fold += model.predict(X_val) / 5
            
        oof_val[:, model_idx] = val_preds_fold
        
    oof_train_df = pd.DataFrame(oof_train, columns=list(base_models.keys()))
    oof_val_df = pd.DataFrame(oof_val, columns=list(base_models.keys()))
    
    print("Entrenando Meta-modelo Ridge en Nivel 1...")
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    meta_model.fit(oof_train_df, y_train)
    
    # Predicciones de Validación para Stacking
    stacking_preds = meta_model.predict(oof_val_df)
    
    # Ajustar modelos base sobre el 100% de X_train para cálculo SHAP posterior
    print("\nAjustando modelos base finales para análisis XAI (SHAP)...")
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        
    # Armar diccionario de predicciones de validación
    y_preds_dict = {
        'RandomForest': rf.predict(X_val),
        'XGBoost': xgb.predict(X_val),
        'LightGBM': lgbm.predict(X_val),
        'Stacking': stacking_preds
    }
    
    # 1. Generar e iniciar gráfico Predicho vs Real
    pred_vs_act_path = os.path.join(figures_dir, "predicted_vs_actual.png")
    plot_predicted_vs_actual(y_val, y_preds_dict, save_path=pred_vs_act_path)
    
    # 2. Generar e iniciar gráfico de Importancias SHAP y coeficientes Ridge
    plot_feature_importances(base_models, meta_model, X_train, X_val, features, save_dir=figures_dir)
    
    print("\n[ÉXITO] Pipeline de visualización y explicabilidad ejecutado correctamente.")

if __name__ == "__main__":
    main()
