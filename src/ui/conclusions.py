import streamlit as st

def render_conclusions_page() -> None:
    st.markdown("<h1 style='text-align: center; color: #1e3a8a; font-weight: 800; font-size: 2.5rem; margin-bottom: 2rem;'>🏁 Conclusión y Próximos Pasos</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .conclusion-card {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            padding: 2rem;
            border-radius: 12px;
            border-left: 5px solid #10b981;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .step-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        .step-icon {
            font-size: 2.2rem;
            margin-right: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="conclusion-card">
        <h2 style='color: #047857; margin-top: 0;'>✅ Conclusión Empírica</h2>
        <p style='font-size: 1.15rem; color: #065f46; line-height: 1.6; margin-bottom: 0;'>
        Se ha comprobado empíricamente que la arquitectura híbrida de <strong>Stacking Dinámico con Hard Routing (6 Clústeres)</strong> es inmensamente superior a los enfoques tradicionales. Al unificar la especialización local (MoE) con la generalización global (Stacking Lineal), logramos dos herramientas de grado productivo: <strong>El Escudo de Varianza (Ridge)</strong> para finanzas macro y <strong>El Francotirador (Huber)</strong> para predicciones exactas de cara al usuario.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #334155; margin-top: 2rem; margin-bottom: 1.5rem;'>🚀 Próximos Pasos</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">🔬</div>
        <div>
            <h4 style='margin: 0; color: #1e293b; font-size: 1.1rem;'>Explicabilidad (SHAP Values)</h4>
            <p style='margin: 0; color: #475569; margin-top: 0.3rem;'>Diseñar un pipeline personalizado que ejecute SHAP localmente para los 6 regímenes de mercado, permitiendo auditar la importancia de las variables a nivel de cada alojamiento individual.</p>
        </div>
    </div>
    
    <div class="step-card">
        <div class="step-icon">🗺️</div>
        <div>
            <h4 style='margin: 0; color: #1e293b; font-size: 1.1rem;'>Análisis Profundo de Clústeres</h4>
            <p style='margin: 0; color: #475569; margin-top: 0.3rem;'>Realizar una inmersión comercial para descifrar exactamente qué perfil de hoteles, amenities o zonas geográficas agrupó de manera autónoma el Árbol de Decisión en sus 6 hojas.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
