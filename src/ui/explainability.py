import streamlit as st

def render_explainability_page() -> None:
    """Renders the Explainability (SHAP) page as Future Research."""
    
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Auditoría de Decisiones y Explicabilidad (SHAP)</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #7f8c8d; margin-bottom: 2em;'>Fase 2: Transparencia Algorítmica</h4>", unsafe_allow_html=True)
    
    st.info("""
    **Estado Actual:** En fase de Investigación y Desarrollo (Próximos Pasos).
    """)
    
    st.markdown("""
    ### 🚧 Arquitectura de Caja Blanca en Desarrollo
    
    El objetivo fundamental de evitar arquitecturas de "caja negra" es permitir la auditoría de las decisiones algorítmicas a nivel de observaciones individuales. Para ello, la técnica estándar en la industria es el cálculo de **valores SHAP (SHapley Additive exPlanations)**.
    
    Sin embargo, nuestra actual arquitectura ganadora (**Stacking Dinámico con Hard Routing**) presenta un desafío topológico extremo para los frameworks tradicionales de explicabilidad:
    
    1. **Fragmentación Espacial:** El *Decision Tree* enruta los datos hacia 6 clústeres distintos.
    2. **Multiplicidad de Expertos:** Dentro de cada clúster, coexisten múltiples modelos (Tweedie XGBoost, Tweedie LightGBM, MAE LightGBM).
    3. **Juez Lineal:** El resultado final es una combinación continua ponderada por un Meta-Modelo (Huber/Ridge).
    
    #### 🔬 Próximos Pasos e Investigaciones Futuras
    Para alcanzar la plena madurez del proyecto, la Fase 2 contempla los siguientes hitos de desarrollo (tal como se estipula en el Reporte Oficial):
    
    1. **Explicabilidad (SHAP):** Diseñar un pipeline personalizado que ejecute SHAP localmente y por duplicado para cada uno de los 6 clústeres, y luego concatene los valores multiplicándolos por los coeficientes del meta-modelo.
    2. **Análisis Profundo de Clústeres:** Analizar a nivel comercial el contenido exacto de los 6 clústeres generados por el enrutador espacial, determinando qué tipos de hoteles, amenities o franjas geográficas capturó naturalmente el árbol de decisión.
    
    Este desarrollo arquitectónico se encuentra planificado para la siguiente etapa operativa.
    """)
    
    st.divider()
    
    st.markdown("### 🔍 Auditoría de Precios por Alojamiento")
    st.markdown("Seleccione un hotel del set de validación (Caja Fuerte) para contrastar el precio real empírico frente a la tasación calculada por nuestros dos modelos ganadores.")
    
    import os
    import pandas as pd
    
    # Resolver ruta absoluta independientemente desde dónde se llame
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, "data", "cache", "test_predictions.csv")
    
    if os.path.exists(csv_path):
        try:
            df_preds = pd.read_csv(csv_path)
            
            # Limpiar hoteles sin nombre o repetidos
            df_preds = df_preds.dropna(subset=['hotel'])
            df_preds = df_preds[df_preds['hotel'].str.strip() != ""]
            
            hotel_list = df_preds['hotel'].unique().tolist()
            
            # Filtrar si hay muchisimos hoteles para no tildar la UI
            if len(hotel_list) > 1000:
                hotel_list = hotel_list[:1000] 
                
            selected_hotel = st.selectbox("🏨 Seleccione un Hotel:", hotel_list)
            
            if selected_hotel:
                hotel_data = df_preds[df_preds['hotel'] == selected_hotel].iloc[0]
                
                real_price = hotel_data['precio_real']
                huber_price = hotel_data['prediccion_huber']
                ridge_price = hotel_data['prediccion_ridge']
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Precio Real Observado", f"${real_price:,.2f} USD")
                
                delta_huber = huber_price - real_price
                col2.metric("El Francotirador (Huber)", f"${huber_price:,.2f} USD", f"{delta_huber:,.2f} USD", delta_color="inverse")
                
                delta_ridge = ridge_price - real_price
                col3.metric("El Escudo (Ridge)", f"${ridge_price:,.2f} USD", f"{delta_ridge:,.2f} USD", delta_color="inverse")
                
                st.info("💡 **Nota:** Valores en rojo o verde (Deltas) indican el sobreprecio o subprecio tasado por el modelo respecto a la realidad empírica.")
                
        except Exception as e:
            st.error(f"Error al leer predicciones: {e}")
    else:
        st.warning("⚠️ **Predicciones no encontradas.** Por favor, ejecute el script `train_stacking.py` para generar y cachear las predicciones del modelo en la carpeta `data/cache/test_predictions.csv`.")

    
    st.divider()
    st.markdown("<br><p style='text-align: center; font-size: 0.8em; color: gray;'>Reporte generado para propósitos de auditoría y revisión arquitectónica.</p>", unsafe_allow_html=True)
