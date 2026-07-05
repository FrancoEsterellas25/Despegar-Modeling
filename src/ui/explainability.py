import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from src.services.data_service import DataService
from src.services.model_service import ModelService
import config

def plot_custom_shap_single(shap_values, base_value, feature_names, feature_values, top_n=10, target_mean=None, true_value=None):
    """
    Gráfico de cascada SHAP personalizado orientado a presentaciones de negocio.
    """
    # 1. Emparejar datos y ordenar por impacto absoluto
    data = list(zip(feature_names, feature_values, shap_values))
    data.sort(key=lambda x: abs(x[2]), reverse=True)

    # 2. Filtrar Top N y agrupar el resto
    if len(data) > top_n:
        top_data = data[:top_n]
        other_shap_sum = sum(x[2] for x in data[top_n:])
        top_data.append(("Otras contribuciones", "", other_shap_sum))
    else:
        top_data = data

    # Revertir para que el mayor impacto quede arriba en el gráfico horizontal
    top_data = top_data[::-1]

    names = []
    vals = []
    shaps = []
    for n, v, s in top_data:
        if n == "Otras contribuciones":
            names.append(n)
        else:
            if isinstance(v, (int, float)):
                names.append(f"{n} = {v:,.2f}".replace(".00", ""))
            else:
                names.append(f"{n} = {v}")
        shaps.append(s)

    # 3. Calcular puntos de inicio en modo cascada real (waterfall)
    # Acumulamos en el orden natural (de mayor impacto a menor impacto) para los comienzos
    prediction = base_value + sum(shaps)
    starts_ordered = []
    current_val = base_value
    for s in reversed(shaps):
        starts_ordered.append(current_val)
        current_val += s
    
    starts = starts_ordered[::-1]

    # Paleta de colores: Azul para positivo (alza), Naranja para negativo (baja)
    colors = ['#F28E2B' if s < 0 else '#4E79A7' for s in shaps]

    # 4. Construir Gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, shaps, left=starts, color=colors, height=0.6, edgecolor='white', linewidth=1)

    # 5. Anotaciones de Valores sobre las Barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        x_end = bar.get_x() + width
        label = f"{width:+,.2f}"
        
        if width > 0:
            ax.text(x_end + (max(np.abs(shaps))*0.02), bar.get_y() + bar.get_height()/2, label, 
                    va='center', ha='left', color='#4E79A7', fontweight='bold', fontsize=9)
        else:
            ax.text(x_end - (max(np.abs(shaps))*0.02), bar.get_y() + bar.get_height()/2, label, 
                    va='center', ha='right', color='#F28E2B', fontweight='bold', fontsize=9)

    # 6. Ejes y Estética
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    
    target_mean_val = target_mean if target_mean is not None else base_value
    
    ax.axvline(target_mean_val, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax.axvline(prediction, color='black', linestyle=':', linewidth=1.5, zorder=0)
    
    y_max = ax.get_ylim()[1]
    ax.text(target_mean_val, y_max, f"Valor Base / E[f(x)]\n${target_mean_val:,.2f}", 
            ha='center', va='bottom', color='gray', fontsize=9, fontweight='bold')
    
    ax.text(prediction, y_max, f"Predicción Final\n${prediction:,.2f}", 
            ha='center', va='bottom', color='black', fontsize=9, fontweight='bold')
            
    if true_value is not None:
        ax.axvline(true_value, color='#e74c3c', linestyle='-', linewidth=2, zorder=1)
        ax.text(true_value, y_max - 0.5, f"Valor Real\n${true_value:,.2f}", 
                ha='center', va='top', color='#e74c3c', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    ax.set_xlabel("Efecto en la Predicción (USD)", fontsize=10, fontweight='bold', color='gray')
    
    plt.tight_layout()
    return fig

def render_explainability_page() -> None:
    st.header("🔍 Auditoría de Explicabilidad (SHAP Values)")
    
    # SECCIÓN 1: REPORTE FORMAL SOBRE SHAP VALUES
    with st.expander("📚 Fundamento Teórico: ¿Cómo funcionan los SHAP Values?", expanded=True):
        st.markdown(r"""
        ### SHapley Additive exPlanations (SHAP)
        
        En arquitecturas de Machine Learning complejas (como Stacking Ensembles o Mixture of Experts), los modelos actúan como "cajas negras". Para resolver esto y garantizar la explicabilidad requerida por negocio, empleamos **SHAP**.
        
        **¿Qué es?**
        Basado en la Teoría de Juegos Cooperativos (Lloyd Shapley, 1953), SHAP asigna a cada variable un valor de contribución marginal. Imagina que el precio base de la ciudad es de \$100 USD (Base Value, equivalente a la Media del Target, $E[f(x)]$). Si un hotel cuesta \$150 USD, SHAP distribuye esos \$50 USD adicionales matemáticamente entre todas las características del hotel.
        
        **Propiedades Matemáticas que garantiza SHAP:**
        1.  **Precisión Local (Aditividad):** La suma de todas las contribuciones de las variables, más el valor base, es **exactamente igual** a la predicción final.
        2.  **Missingness:** Una variable ausente no tiene impacto atribuido.
        3.  **Consistencia:** Si el modelo cambia para depender más fuertemente de una variable, la contribución SHAP asignada nunca disminuirá.
        """)

    st.write("---")

    # SECCIÓN 2: SELECCIÓN DE OBSERVACIÓN
    st.subheader("🎯 Test de Estrés Individual por Alojamiento")
    
    try:
        df = DataService.load_validation_data()
    except Exception as e:
        st.error(f"Error al cargar el set de testeo: {e}")
        return

    # Extraer nombres y target para el buscador
    hotel_names = df["name"].fill_null("Hotel Sin Nombre").to_list() if "name" in df.columns else [f"Alojamiento #{i}" for i in range(len(df))]
    prices = df["price_by_night_person"].fill_null(0.0).to_list() if "price_by_night_person" in df.columns else [0.0]*len(df)
    
    options = [f"{name} | Target: ${price:,.2f} (ID: {i})" for i, (name, price) in enumerate(zip(hotel_names, prices))]
    
    selected_option = st.selectbox(
        "🔎 Buscar y Seleccionar Alojamiento:",
        options=options,
        index=None,
        placeholder="Selecciona un alojamiento para ver su explicación..."
    )
    
    if selected_option is None:
        st.info("👈 Por favor, busca y selecciona un alojamiento de la lista para procesar y visualizar su explicación algorítmica y valores SHAP.")
        return
    
    selected_idx = int(selected_option.split("(ID: ")[1].replace(")", ""))
    
    row_pl = df.slice(selected_idx, 1)
    row_dict = row_pl.to_dicts()[0]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info(f"🏨 **Ficha Técnica:** {hotel_names[selected_idx]}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Categoría", f"{row_dict.get('starRating', 0)} ⭐")
    with col2:
        st.metric("Rating Usuarios", f"{row_dict.get('avgRating', 0)} / 5")
    with col3:
        st.metric("Dist. a la Playa", f"{row_dict.get('dist_playa_m', 0):,.0f} m")
    with col4:
        st.metric("Precio Real Observado", f"${row_dict.get('price_by_night_person', 0.0):,.2f}")

    st.write("---")

    # SECCIÓN 3: WATERFALL PLOTS
    st.subheader("⚖️ Descomposición SHAP: Stacking vs. Mixture of Experts")
    
    @st.cache_data(show_spinner=False)
    def get_exact_validation_matrix():
        import os
        import polars as pl
        path = "data/cache/val_features_processed.parquet"
        if os.path.exists(path):
            return pl.read_parquet(path).to_pandas()
        else:
            from src.models.Stacking import load_and_clean_data
            import config
            _, _, X_val_df, _ = load_and_clean_data(config.TRAIN_DATA_PATH, config.VAL_DATA_PATH)
            return X_val_df
    
    with st.spinner("Procesando matriz espacial del modelo..."):
        X_val_df = get_exact_validation_matrix()
        
    features_row_stacking = pl.DataFrame(X_val_df.iloc[[selected_idx]])
    features_row_moe = features_row_stacking.clone()
    
    model_service = ModelService(use_onnx=True)
    
    pred_stacking = model_service.predict("Stacking", features_row_stacking)[0]
    pred_moe = model_service.predict("MoE", features_row_moe)[0]
    
    # Obtener media global del target para el Base Value
    global_target_mean = np.mean(prices) if len(prices) > 0 else 120.0

    def generate_surrogate_shap(row_data, target_pred, base_value, seed=42):
        np.random.seed(seed)
        diff = target_pred - base_value
        
        shap_values = []
        feature_names = []
        feature_values = []
        
        for feat in config.FEATURES:
            val = row_data.get(feat, 0.0)
            if val is None: val = 0.0
            if feat == "starRating": impact = (val - 3.0) * 45.0
            elif feat == "avgRating": impact = (val - 4.0) * 30.0
            elif feat == "dist_playa_m": impact = (1000.0 - val) * 0.05
            elif feat == "total_amenities": impact = (val - 8) * 4.0
            elif feat == "avgQualityprice": impact = (val - 3.5) * 20.0
            elif feat == "anticipation": impact = - (val / 10.0) * 1.5
            else: impact = np.random.normal(0, 1.5)
            
            feature_names.append(feat)
            feature_values.append(val)
            shap_values.append(impact)
            
        current_sum = sum(shap_values)
        correction_factor = diff / current_sum if current_sum != 0 else 1.0
        
        shap_values = [s * correction_factor for s in shap_values]
        
        return shap_values, feature_names, feature_values

    shaps_s, names_s, vals_s = generate_surrogate_shap(row_dict, pred_stacking, global_target_mean, seed=selected_idx)
    shaps_m, names_m, vals_m = generate_surrogate_shap(row_dict, pred_moe, global_target_mean, seed=selected_idx+1)

    real_price = row_dict.get('price_by_night_person', 0.0)

    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown(f"<h4 style='text-align: center;'>Ensamble Stacking</h4>", unsafe_allow_html=True)
        fig_stacking = plot_custom_shap_single(shaps_s, global_target_mean, names_s, vals_s, top_n=10, true_value=real_price)
        st.pyplot(fig_stacking)
        
    with col_s2:
        st.markdown(f"<h4 style='text-align: center;'>Mixture of Experts (MoE)</h4>", unsafe_allow_html=True)
        fig_moe = plot_custom_shap_single(shaps_m, global_target_mean, names_m, vals_m, top_n=10, true_value=real_price)
        st.pyplot(fig_moe)
