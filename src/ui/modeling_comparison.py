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
    Esta sección expone los fundamentos algorítmicos y topológicos de los dos motores de inferencia propuestos. El enfoque *Mixture of Experts (MoE)* apuesta por la hiperespecialización de modelos en subespacios de datos (ej. hoteles de lujo vs. estándar). Por su parte, la arquitectura de *Stacking de 2 Niveles* busca la generalización global combinando predicciones condicionales a través de un meta-estimador paramétrico. A continuación, se detallan las estructuras subyacentes y se presenta la evaluación comparativa empírica sobre el conjunto de validación de exclusión (*hold-out*).
    """)
    st.divider()

    # -------------------------------------------------------------------------
    # ARQUITECTURAS
    # -------------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color: #27ae60;'>I. Topología Mixture of Experts (MoE)</h3>", unsafe_allow_html=True)
        st.markdown("""
        **Filosofía de Diseño:** *Divide et impera* (Divide y vencerás).
        La premisa central asume que el comportamiento del precio de un alojamiento turístico obedece a funciones matemáticas marcadamente diferentes dependiendo de su nicho de mercado (un hotel boutique no escala el precio de sus *amenities* de la misma forma que un resort estándar).

        **Estructura del Grafo Computacional:**
        1.  **Función Gating (Enrutador):** Un clasificador de Gradient Boosting (`XGBClassifier`) actúa como nodo raíz. Evalúa las características cualitativas iniciales (ej. Estrellas, MCA Latente) y asigna un peso probabilístico o una ruta estricta (hard-routing) hacia un experto específico.
        2.  **Expertos Especializados (Experts):** Modelos regresores independientes (`XGBRegressor`) que han sido entrenados **exclusivamente** con el subconjunto de datos de su propio segmento (Estándar, Lujo, Boutique/Informal).
        
        **Ventaja Teórica:**
        Reducción del sesgo condicional. Cada árbol de decisión en el experto no malgasta cortes (splits) intentando separar hoteles de lujo de los baratos, concentrándose únicamente en optimizar la micro-varianza de su segmento.
        """)

    with col2:
        st.markdown("<h3 style='color: #2980b9;'>II. Topología Ensamble Stacking</h3>", unsafe_allow_html=True)
        st.markdown("""
        **Filosofía de Diseño:** Consenso de Expertos Globales.
        En lugar de particionar el conjunto de datos, se aprovecha la diversidad algorítmica. Distintas familias de algoritmos capturan diferentes patrones en el espacio de características global.

        **Estructura del Grafo Computacional:**
        1.  **Nivel 0 (Estimadores Base):** Un conjunto heterogéneo de modelos entrenados de forma paralela sobre la totalidad del dataset:
            *   *Random Forest:* Excelente generalización vía *bagging* y alta robustez ante *outliers*.
            *   *XGBoost:* Descenso de gradiente puro, optimización extrema de funciones de pérdida.
            *   *LightGBM:* Gradient Boosting basado en hojas (leaf-wise growth), alta eficiencia con variables categóricas proyectadas.
        2.  **Nivel 1 (Meta-Modelo):** Un estimador paramétrico lineal simple (`LinearRegression` / `RidgeCV`). Se alimenta estrictamente de las predicciones Out-Of-Fold (OOF) del Nivel 0.
        
        **Ventaja Teórica:**
        Minimización estructural de la varianza. El meta-estimador aprende a ponderar en tiempo real a qué modelo base "creerle" según su confianza implícita, corrigiendo los errores direccionales de los demás.
        """)

    st.divider()

    # -------------------------------------------------------------------------
    # EVALUACIÓN EMPÍRICA (MÉTRICAS)
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Evaluación de Rendimiento Generalización (Set de Validación)")
    
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
            Para auditar de forma justa arquitecturas tan complejas, hemos evaluado los modelos sobre la totalidad del **Dataset de Validación (OOT - Out of Time / Hold-out)** utilizando las siguientes funciones de pérdida:

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
