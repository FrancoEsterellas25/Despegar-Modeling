import streamlit as st
import polars as pl
import pandas as pd

def render_modeling_comparison_page() -> None:
    """Renders the Modeling Comparison report in a formal layout."""
    
    # -------------------------------------------------------------------------
    # ENCABEZADO DEL REPORTE
    # -------------------------------------------------------------------------
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Reporte Técnico: Arquitecturas de Modelado y Evaluación</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #7f8c8d; margin-bottom: 2em;'>Frontera de Pareto: El Escudo de Varianza vs. El Francotirador</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    **Resumen Ejecutivo:**
    Esta sección expone la arquitectura final consolidada: **Stacking Dinámico con Enrutamiento Orientado al Target (Hard Routing)**. 
    Hemos superado las aproximaciones tradicionales (MoE puro o Stacking lineal simple) mediante un pipeline híbrido. 
    Se divide el espacio en 6 regímenes de precio usando un Árbol de Decisión supervisado (sobre variables latentes MCA, geografía y tamaño). Dentro de cada clúster, compiten modelos expertos optimizados con funciones Tweedie y MAE. 
    A continuación, comparamos los dos Meta-Modelos ganadores que ponderan a dichos expertos en la Capa 2.
    """)
    st.divider()

    # -------------------------------------------------------------------------
    # METODOLOGÍA DE EVALUACIÓN DE MODELOS
    # -------------------------------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: #8e44ad; padding: 15px; background-color: #f4ecf7; border-radius: 8px; margin-bottom: 20px; font-weight: bold;'>🔬 Metodología de Evaluación de Modelos</h2>", unsafe_allow_html=True)
    st.info("""
    La robustez de la arquitectura fue probada bajo condiciones de exclusión estricta de *Data Leakage*:
    *   **Partición por Sesión (searchid):** La separación Train/Validation se garantizó aislando las sesiones de búsqueda.
    *   **Caja Fuerte (Test Set):** Evaluación final sobre el 100% de la varianza real (incluyendo outliers millonarios) en un conjunto jamás visto por el modelo.
    *   **Banderas is_missing:** El modelo aprende explícitamente de la ausencia de ratings y estrellas a través de variables booleanas.
    """)
    st.divider()

    # -------------------------------------------------------------------------
    # ARQUITECTURAS GANADORAS
    # -------------------------------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #27ae60;'>I. El Escudo de Varianza</h3>", unsafe_allow_html=True)
        st.markdown("""
        **Tweedie Puro + RIDGE Regression**
        - **Capa 1 (Expertos):** Usa exclusivamente predicciones de los modelos XGBoost y LightGBM optimizados con pérdida Tweedie.
        - **Capa 2 (Juez):** Regresión Ridge (L2).
        - **Objetivo:** Minimizar el Error Cuadrático (MSE).
        - **Uso Recomendado:** Reportes agregados y Revenue Management macro. Evita que la volatilidad de los hoteles de lujo rompa el balance mensual.
        """)

    with col2:
        st.markdown("<h3 style='color: #2980b9;'>II. El Francotirador</h3>", unsafe_allow_html=True)
        st.markdown("""
        **Tweedie & MAE + HUBER Regressor**
        - **Capa 1 (Expertos):** Inyecta a la matriz un Experto LightGBM entrenado exclusivamente para optimizar el MAE.
        - **Capa 2 (Juez):** Regresión de Huber (Transición suave entre MSE y MAE).
        - **Objetivo:** Minimizar el Error Absoluto en dólares.
        - **Uso Recomendado:** Front-End de Despegar (usuario final). Ignora a los millonarios atípicos y clava la estimación exacta para el usuario promedio.
        """)

    st.divider()

    # Métricas consolidadas del backend
    metrics_data = [
        {"Arquitectura": "Regresión Lineal Simple (Baseline)", "MAE (USD)": "$50.94", "R²": "0.1250", "Estado": "Descartado ❌"},
        {"Arquitectura": "El Escudo (Tweedie + Ridge)", "MAE (USD)": "$22.87", "R²": "0.6766", "Estado": "Ganador Varianza 🏆"},
        {"Arquitectura": "El Francotirador (Tweedie+MAE + Huber)", "MAE (USD)": "$19.42", "R²": "0.6131", "Estado": "Ganador Precisión 🏆"}
    ]
    metrics_df = pd.DataFrame(metrics_data)
    
    st.markdown("### 📊 Evaluación de Rendimiento Empírico (Set de Testeo sin filtrar)")
    st.dataframe(metrics_df, use_container_width=True)
    
    st.info("💡 **Justificación de Complejidad:** Como se observa en la tabla superior, asumir un comportamiento lineal y global para todo Río de Janeiro (Baseline) produce un error inadmisible de **$50.94 USD** por noche. La segmentación espacial y el ensamble de expertos de nuestra arquitectura reduce ese error en más de 31 dólares, justificando plenamente el desarrollo algorítmico.")
    
    st.markdown("""
    > **💡 Nota sobre Limpieza de Datos:** En un experimento paralelo (truncando el dataset para ocultar los outliers $>1000$ USD), **El Francotirador** llegó a un récord artificial de **$16.23 USD** de MAE, mientras que **El Escudo** superó el **0.71** de $R^2$. Sin embargo, los modelos expuestos arriba (evaluados sobre la totalidad del mercado sin truncar) son los que poseen la verdadera madurez para producción.
    """)
    
    # -------------------------------------------------------------------------
    # EXPLICACIÓN TEÓRICA Y MATEMÁTICA DE MÉTRICAS (LATEX)
    # -------------------------------------------------------------------------
    with st.expander("📚 Glosario de Métricas Matemáticas y Criterios de Evaluación", expanded=False):
        st.markdown(r"""
        Para auditar de forma justa arquitecturas tan complejas, evaluamos los modelos sobre la totalidad del **Dataset de Testeo (OOT - Out of Time / Hold-out)** utilizando las siguientes funciones:

        **1. Mean Absolute Error (MAE)**  
        Mide el promedio de los errores en valor absoluto. Al no estar elevado al cuadrado, es sumamente robusto a valores atípicos y se interpreta directamente como el desvío promedio en dólares que vería un cliente.
        $$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

        **2. Coeficiente de Determinación ($R^2$)**  
        Mide la proporción de la varianza del precio real que es matemáticamente explicada por el modelo. Es esclavo de las colas pesadas (outliers), por lo que un $R^2$ alto en este negocio hipervolátil indica una estabilidad sistémica excepcional.
        $$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$
        """)
