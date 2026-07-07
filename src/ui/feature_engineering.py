import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from src.services.data_service import DataService

def render_feature_engineering_page() -> None:
    """Renders the Feature Engineering report in a formal, modular layout."""
    
    # -------------------------------------------------------------------------
    # ENCABEZADO DEL REPORTE
    # -------------------------------------------------------------------------
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Reporte Técnico: Ingeniería de Características (Feature Engineering)</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #7f8c8d; margin-bottom: 2em;'>Metodología de Extracción y Transformación de Variables</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    **Resumen Ejecutivo:**
    El presente documento detalla la metodología aplicada para la construcción, transformación y selección de variables predictoras subyacentes a los modelos de *Mixture of Experts (MoE)* y *Stacking*. El diseño del pipeline de Feature Engineering tiene como objetivo maximizar la varianza explicada del precio de los alojamientos en Río de Janeiro, mitigando el sobreajuste (overfitting) y capturando relaciones complejas, tanto espaciales como temporales.
    """)
    st.divider()

    # Intentar cargar datos de testeo
    try:
        df = DataService.load_validation_data()
        data_available = True
    except Exception:
        df = None
        data_available = False

    # -------------------------------------------------------------------------
    # SECCIÓN 1: MCA
    # -------------------------------------------------------------------------
    st.markdown("### 1. Representación de Espacio Latente: Multiple Correspondence Analysis (MCA)")
    
    col1_mca, col2_mca = st.columns([3, 2])
    with col1_mca:
        st.markdown("""
        **Fundamento Matemático y Propósito:**
        La variable de comodidades (*amenities*) presenta una cardinalidad extremadamente alta y una estructura dispersa (sparse). Para mitigar la maldición de la dimensionalidad, se implementó un Análisis de Correspondencias Múltiples (MCA).
        
        **Implementación Estricta y Fuga de Información (Data Leakage):**
        *   **Entrenamiento:** El algoritmo MCA se ajusta **exclusivamente utilizando el conjunto de datos de entrenamiento**. Esto genera un espacio de menor dimensionalidad que preserva la mayor cantidad de inercia (varianza) posible de las categorías originales.
        *   **Proyección:** Las observaciones pertenecientes al conjunto de **testeo son estrictamente proyectadas** sobre este espacio euclidiano ya aprendido. En ningún escenario las observaciones de testeo modifican los pesos o la estructura del espacio latente, garantizando una evaluación insesgada.
        *   **Resultado:** Cada alojamiento es representado mediante coordenadas continuas (`mca_dim_1` a `mca_dim_8`), transformando atributos cualitativos en vectores numéricos densos.
        """)
    with col2_mca:
        if data_available and "mca_dim_1" in df.columns:
            st.markdown("<p style='text-align: center; font-size: 0.9em; color: gray;'>Proyección de Testeo en Espacio MCA (Dim 1 vs Dim 2)</p>", unsafe_allow_html=True)
            fig_mca, ax_mca = plt.subplots(figsize=(6, 4))
            # Usando datos REALES ahora que fueron calculados correctamente
            sns.scatterplot(x=df["mca_dim_1"].to_numpy(), y=df["mca_dim_2"].to_numpy(), alpha=0.4, color="#3498db", edgecolor=None, s=20, ax=ax_mca)
            ax_mca.set_xlabel("MCA Componente 1")
            ax_mca.set_ylabel("MCA Componente 2")
            st.pyplot(fig_mca, use_container_width=True)
        else:
            st.info("Visualización no disponible (datos no cargados).")

    st.divider()

    # -------------------------------------------------------------------------
    # SECCIÓN 2: GEOESPACIAL
    # -------------------------------------------------------------------------
    st.markdown("### 2. Ingeniería Geoespacial y Distancias Ortodrómicas")
    
    col1_geo, col2_geo = st.columns([3, 2])
    with col1_geo:
        st.markdown("""
        **Modelado del Contexto Espacial:**
        El precio de los bienes raíces y servicios de hospitalidad exhibe una fuerte dependencia espacial. Se construyeron métricas topológicas para cuantificar esta influencia.
        
        *   **Distancia de Haversine:** Se calculó la distancia esférica exacta desde las coordenadas (Latitud, Longitud) de cada hotel hacia una matriz de Puntos de Interés (POIs) críticos en Río de Janeiro, tales como aeropuertos (GIG, SDU), playas principales (Copacabana, Ipanema) y monumentos (Cristo Redentor).
        *   **Variables de Entorno Socioeconómico:** Se incorporaron distancias métricas hacia zonas de vulnerabilidad (favelas) junto con atributos de dichas zonas (tamaño, densidad de viviendas).
        *   **Estimación de Densidad por Kernel (KDE):** Se calculó un estimador de densidad para medir el nivel de aglomeración turística (competencia) en las inmediaciones del alojamiento.
        """)
    with col2_geo:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Insights Geoespaciales (Muestra de Testeo)**")
        if data_available and "dist_playa_m" in df.columns:
            # En lugar del histograma, presentamos métricas de resumen elegantes
            playa_mediana = df["dist_playa_m"].median()
            playa_min = df["dist_playa_m"].min()
            playa_p90 = df["dist_playa_m"].quantile(0.90)
            
            st.metric(label="Mediana de Distancia a la Costa", value=f"{playa_mediana:,.0f} metros")
            
            st.markdown(f"""
            - **Mayor proximidad:** {playa_min:,.0f} m.
            - **Cobertura Densa (90%):** A menos de {playa_p90:,.0f} m. de la costa.
            
            *La centralidad espacial demuestra una altísima correlación entre la oferta hotelera y el litoral costero (Copacabana, Ipanema, Barra).*
            """)
        else:
            st.warning("Datos geoespaciales no disponibles.")

    st.divider()

    # -------------------------------------------------------------------------
    # SECCIÓN 3: TEMPORALIDAD
    # -------------------------------------------------------------------------
    st.markdown("### 3. Modelado Cíclico Temporal y Detección de Eventos")
    
    col1_temp, col2_temp = st.columns([2, 3])
    with col1_temp:
        if data_available and "wday_ci_sine" in df.columns and "wday_ci_cosine" in df.columns:
            st.markdown("<p style='text-align: center; font-size: 0.9em; color: gray;'>Espacio Cíclico: Días de la Semana</p>", unsafe_allow_html=True)
            fig_temp, ax_temp = plt.subplots(figsize=(4, 4))
            # Graficamos una submuestra real para no sobrecargar el renderizado (scatter de 270k ptos idénticos es lento)
            sample_df = df.sample(n=min(5000, len(df)))
            sns.scatterplot(x=sample_df["wday_ci_sine"].to_numpy(), y=sample_df["wday_ci_cosine"].to_numpy(), color="#9b59b6", s=100, edgecolor="w", ax=ax_temp, zorder=5)
            circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.5)
            ax_temp.add_patch(circle)
            ax_temp.set_xlabel("Seno (Día de la semana)")
            ax_temp.set_ylabel("Coseno (Día de la semana)")
            ax_temp.axhline(0, color='grey', linestyle='-', linewidth=0.5)
            ax_temp.axvline(0, color='grey', linestyle='-', linewidth=0.5)
            ax_temp.set_xlim(-1.2, 1.2)
            ax_temp.set_ylim(-1.2, 1.2)
            ax_temp.set_aspect('equal', adjustable='box')
            st.pyplot(fig_temp, use_container_width=True)
        else:
            st.info("Visualización no disponible.")
            
    with col2_temp:
        st.markdown("""
        **Codificación Trigonométrica:**
        Las variables relacionadas con el calendario (por ejemplo, el día de la semana del check-in o el mes) son inherentemente cíclicas (el domingo es adyacente al lunes). Representarlas como variables ordinales lineales introduce sesgos. Por lo tanto, se transformaron en un espacio euclidiano continuo mediante funciones seno y coseno, preservando la topología circular del tiempo.
        
        **Indicadores Estacionales y Feriados:**
        La dinámica de precios en Río de Janeiro está severamente condicionada por eventos anómalos. Se construyeron variables indicadoras booleanas (Flags) para aislar el impacto de la demanda extraordinaria:
        *   Carnaval (2024, 2025)
        *   Reveillon (Año Nuevo)
        *   Rock in Rio
        """)

    st.divider()


        
    st.markdown("<br><p style='text-align: center; font-size: 0.8em; color: gray;'>Reporte generado para propósitos de auditoría y revisión arquitectónica.</p>", unsafe_allow_html=True)
