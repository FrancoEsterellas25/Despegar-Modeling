import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.services.model_service import ModelService
from src.services.data_service import DataService

def render_modeling_comparison_page() -> None:
    """Renders the Modeling Comparison report in a formal layout."""
    
    # -------------------------------------------------------------------------
    # ENCABEZADO DEL REPORTE
    # -------------------------------------------------------------------------
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Reporte Técnico: Arquitecturas de Modelado y Evaluación</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #7f8c8d; margin-bottom: 2em;'>Comparativa Estratégica: Mixture of Experts vs. Ensamble Stacking</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    **Resumen Ejecutivo:**
    Esta sección expone los fundamentos algorítmicos y topológicos de los dos motores de inferencia propuestos. El enfoque *Mixture of Experts (MoE)* apuesta por la hiperespecialización de modelos en subespacios de datos (ej. hoteles de lujo vs. estándar). Por su parte, la arquitectura de *Stacking de 2 Niveles* busca la generalización global combinando predicciones condicionales a través de un meta-estimador paramétrico. A continuación, se detallan las estructuras subyacentes y se presenta la evaluación comparativa empírica sobre el conjunto de testeo de exclusión (*hold-out*).
    """)
    st.divider()

    # -------------------------------------------------------------------------
    # METODOLOGÍA DE EVALUACIÓN DE MODELOS
    # -------------------------------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: #8e44ad; padding: 15px; background-color: #f4ecf7; border-radius: 8px; margin-bottom: 20px; font-weight: bold;'>🔬 Metodología de Evaluación de Modelos</h2>", unsafe_allow_html=True)
    st.info("""
    La metodología empleada para evaluar la robustez de las arquitecturas predictivas se divide en las siguientes etapas:

    *   **Preparación en el Set de Entrenamiento (Train):** Se diseñan las variables explicativas (*Feature Engineering*) y se extraen los estadísticos y parámetros necesarios (ej. hiperplanos del MCA, métricas geoespaciales KNN) que luego se proyectarán de forma estricta sobre el **Set de Testeo**.
    *   **Finetuning (Exclusivo de Ensamble Stacking):** Los estimadores base se optimizan sobre el **Set de Entrenamiento** utilizando validación cruzada (*Cross-Validation*). Las predicciones Out-of-Fold alimentan finalmente al meta-modelo (Regresión Lineal simple), garantizando que aprenda de predicciones no sesgadas.
    *   **Hiper-parametrización de Expertos (Exclusivo de Mixture of Experts):** El espacio muestral se segmenta heurísticamente y cada nicho entrena su propio modelo especializado (**XGBRegressor**) utilizando parámetros optimizados estáticos y detención temprana (*early-stopping*).
    *   **Evaluación Final en el Set de Testeo:** Se proyectan todas las transformaciones aprendidas sobre el **Set de Testeo** (datos nunca antes vistos) y se auditan los resultados utilizando las 5 métricas de negocio estipuladas.
    """)
    st.divider()

    # -------------------------------------------------------------------------
    # ARQUITECTURAS
    # -------------------------------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #27ae60;'>I. Topología Mixture of Experts (MoE)</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Enrutador:** Reglas heurísticas de negocio (basadas en estrellas, cantidad de habitaciones y amenities) que segmentan el inventario en tres grupos: *Lujo/Resort*, *Boutique/Informal* y *Estándar*. 
        - **Expertos:** Modelos XGBoost independientes y especializados (Finetuneados en el espacio logarítmico del target) para cada segmento de negocio.
        """)

    with col2:
        st.markdown("<h3 style='color: #2980b9;'>II. Topología Ensamble Stacking</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Nivel 0 (Base):** Modelos basados en árboles (XGBoost, LightGBM, RandomForest).
        - **Análisis de Correlación de Predicciones Out of Sample:** Entrenamos modelos y mediante CV (Cross-Validation), predecimos un subset de los datos de entrenamiento, con esas predicciones (para los 3 modelos base), calculamos la correlación entre los mismos. Esperamos baja correlación entre las predicciones de los modelos, para no tener modelos que, básicamente, estén diciendo lo mismo. 
        - **Nivel 1 (Meta-modelo):** Regresión Lineal Ordinaria (OLS). Combina las predicciones base para optimizar el error general sin pérdida de linealidad. Además de recibir las primeras 2 componentes principales del PCA de los datos de entrenamiento, a forma de contextualizar al modelo, sin romper el principio de que el meta-modelo solo aprenda combinaciones lineales de las predicciones de los modelos base.
        """)

    st.divider()

    # -------------------------------------------------------------------------
    # EVALUACIÓN EMPÍRICA (MÉTRICAS)
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Evaluación de Rendimiento Generalización (Set de Testeo)")
    
    try:
        import os
        import sys
        import json
        import subprocess
        
        metrics_path = "data/cache/metrics_cache.json"
        preds_path = "data/cache/preds_cache.npz"
        script_path = "get_metrics_models.py"
        
        if not os.path.exists(metrics_path) or not os.path.exists(preds_path):
            with st.spinner("Precalculando métricas (esto puede demorar la primera vez)..."):
                result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"Error calculando métricas:\n{result.stderr}")
                    return
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            results_list = json.load(f)
            
        data_cache = np.load(preds_path)
        y_true = data_cache["y_true"]
        preds_stacking = data_cache["preds_stacking"]
        preds_moe = data_cache["preds_moe"]
        
        # Formatear las métricas directamente en Polars para evitar dependencias de Pandas (Jinja2)
        n_rmse_col = f"n-RMSE (min[{y_true.min():.2f}]-max[{y_true.max():.2f}])"
        formatted_list = []
        for row in results_list:
            if row.get("n-RMSE") is not None and str(row.get("n-RMSE")) != "N/A":
                formatted_list.append({
                    "Arquitectura": row["Arquitectura"],
                    "MAE": f"${row['MAE']:,.2f}",
                    "RMSE": f"${row['RMSE']:,.2f}",
                    n_rmse_col: f"{row['n-RMSE']:.4f}",
                    "p-MAE (%)": f"{row['p-MAE (%)']:.2f}%",
                    "R²": f"{row['R²']:.4f}"
                })
            else:
                formatted_list.append({
                    "Arquitectura": row["Arquitectura"],
                    "MAE": "N/A",
                    "RMSE": "N/A",
                    n_rmse_col: "N/A",
                    "p-MAE (%)": "N/A",
                    "R²": "N/A"
                })
                
        metrics_df = pl.DataFrame(formatted_list)
        
        # Presentación tabular nativa de Streamlit convirtiendo a Pandas plano para evitar el warning de ancho de contenedor de Polars
        st.dataframe(metrics_df.to_pandas(), use_container_width=True)
        
        # -------------------------------------------------------------------------
        # EXPLICACIÓN TEÓRICA Y MATEMÁTICA DE MÉTRICAS (LATEX)
        # -------------------------------------------------------------------------
        with st.expander("📚 Glosario de Métricas Matemáticas y Criterios de Evaluación", expanded=False):
            st.markdown(r"""
            Para auditar de forma justa arquitecturas tan complejas, hemos evaluado los modelos sobre la totalidad del **Dataset de Testeo (OOT - Out of Time / Hold-out)** utilizando las siguientes funciones de pérdida:

            **1. Root Mean Squared Error (RMSE)**  
            Representa el error absoluto promedio de la predicción penalizando de manera cuadrática los errores más grandes (outliers). Se mide en la misma unidad de la variable objetivo (Dólares).
            $$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

            **2. Normalized RMSE (n-RMSE)**  
            Permite medir la dispersión del error de forma adimensionada ajustándolo a la volatilidad real del mercado. En este entorno, normalizamos dividiendo el RMSE por el rango de precios observables (Max - Min).
            $$ n\text{-}RMSE = \frac{RMSE}{y_{max} - y_{min}} $$
            
            **3. Mean Absolute Error (MAE)**  
            Mide el promedio de los errores en valor absoluto. Al no estar elevado al cuadrado, es más robusto a valores atípicos que el RMSE y se interpreta directamente como el desvío promedio en dólares.
            $$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$
            
            **4. Percentage Mean Absolute Error (p-MAE)**  
            A diferencia del MAPE (que es sensible a divisiones por cero o valores cercanos a cero), el p-MAE mide el error porcentual global dividiendo el MAE por el Precio Medio de la Población. Muy útil para el negocio porque indica el desvío promedio en porcentaje.
            $$ p\text{-}MAE = \frac{\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|}{\bar{y}} \times 100 $$

            **5. Coeficiente de Determinación ($R^2$)**  
            Mide la proporción de la varianza del precio real que es matemáticamente explicada por el modelo. Un valor de $1.0$ indica predicciones perfectas, y un valor negativo indicaría que es peor que predecir siempre el precio promedio.
            $$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$
            """)
        
        st.markdown("<br><h4 style='text-align: center; color: #34495e;'>Diagnóstico de Residuos: Predicción vs. Observación Empírica</h4>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gráfico Stacking
        sns.scatterplot(x=y_true, y=preds_stacking, alpha=0.3, color="#2980b9", edgecolor=None, s=15, ax=ax[0])
        ax[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='#e74c3c', linestyle='--', linewidth=2)
        ax[0].set_title("Topología Stacking", fontweight='bold', pad=10)
        ax[0].set_xlabel("Precio Real (Ground Truth)")
        ax[0].set_ylabel("Valor Estimado")
        
        # Gráfico MoE
        sns.scatterplot(x=y_true, y=preds_moe, alpha=0.3, color="#27ae60", edgecolor=None, s=15, ax=ax[1])
        ax[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='#e74c3c', linestyle='--', linewidth=2)
        ax[1].set_title("Topología MoE", fontweight='bold', pad=10)
        ax[1].set_xlabel("Precio Real (Ground Truth)")
        ax[1].set_ylabel("Valor Estimado")
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Falla de inicialización de métricas empíricas: {e}")
