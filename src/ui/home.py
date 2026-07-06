import streamlit as st

def render_home_page() -> None:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a; font-weight: 800; font-size: 2.5rem; margin-bottom: 0px;'>🏨 Modelado de Precios de Alojamiento</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #64748b; font-weight: 400; margin-top: 0px; margin-bottom: 2rem;'>Rio de Janeiro, Brasil 🇧🇷</h3>", unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .hero-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 2rem;
            border-radius: 12px;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .strategy-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            height: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-card">
        <h2 style='color: #1e40af; margin-top: 0;'>🎯 1. Resumen Ejecutivo</h2>
        <p style='font-size: 1.1rem; color: #334155; line-height: 1.6;'>
        Este proyecto tiene como objetivo predecir el <strong>precio por noche por persona</strong> (<code>price_by_night_person</code>) de alojamientos en Rio de Janeiro. Para ello, se ha desarrollado un <strong>pipeline de Machine Learning escalable e interpretable</strong> que permite al negocio entender los factores que impulsan el precio, utilizando datos enriquecidos (características del hotel, amenities, ubicación, estacionalidad, etc.).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #334155; margin-bottom: 1rem;'>🧠 Estrategias de Modelado Experimentales</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="strategy-card">
            <h4 style='color: #0f172a; margin-top: 0;'>🔹 Topologías Divergentes</h4>
            <p style='color: #475569;'>Al mismo tiempo, probamos <b>dos estrategias de modelado distintas</b> que, a su vez, para el mismo conjunto deciden hacer uso diferente del conjunto de datos otorgado.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="strategy-card">
            <h4 style='color: #0f172a; margin-top: 0;'>🔹 Especialización y Agrupamiento</h4>
            <p style='color: #475569;'>Estos enfoques dividen en distintos subsets de datos y entrenan modelos bajo <b>distintos enfoques de agrupamientos</b> (segmentación local vs. generalización global).</p>
        </div>
        """, unsafe_allow_html=True)
