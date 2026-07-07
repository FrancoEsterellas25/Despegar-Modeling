import streamlit as st
from src.ui.home import render_home_page
from src.ui.feature_engineering import render_feature_engineering_page
from src.ui.modeling_comparison import render_modeling_comparison_page
from src.ui.explainability import render_explainability_page
from src.ui.conclusions import render_conclusions_page

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Modelado de Precios de Alojamiento (Rio de Janeiro)",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Diseño estético premium con CSS customizado
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main() -> None:
    """Main function driving the Streamlit application routing."""
    st.sidebar.title("🔮 Modelado de Precios de Alojamiento")
    st.sidebar.write("Río de Janeiro, Brasil")
    st.sidebar.markdown("---")

    # Selección de Módulo
    page = st.sidebar.radio(
        "Navegación:",
        ["Inicio", "Feature Engineering", "Enfoques de Modelado", "Evaluación & SHAP", "Conclusiones"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**📚 Documentación Oficial:**\n\n"
        "Podés consultar el desarrollo matemático y metodológico exhaustivo en el reporte final:\n\n"
        "👉 [Leer Reporte (PDF)](reporte/reporte.pdf)"
    )

    # Ruteo lógico
    if page == "Inicio":
        render_home_page()
    elif page == "Feature Engineering":
        render_feature_engineering_page()
    elif page == "Enfoques de Modelado":
        render_modeling_comparison_page()
    elif page == "Evaluación & SHAP":
        render_explainability_page()
    elif page == "Conclusiones":
        render_conclusions_page()

if __name__ == "__main__":
    main()
