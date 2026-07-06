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
        Se ha comprobado empíricamente que <strong>ambos enfoques (MoE y Stacking) son fructíferos y competitivos</strong>. Dependiendo del caso individual o segmento de mercado, un enfoque predice puntualmente mejor que el otro, pero en términos agregados ambos ostentan una performance muy similar y altamente satisfactoria para el negocio.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: #334155; margin-top: 2rem; margin-bottom: 1.5rem;'>🚀 Próximos Pasos</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-card">
        <div class="step-icon">🧱</div>
        <div>
            <h4 style='margin: 0; color: #1e293b; font-size: 1.1rem;'>Expandir el Ensamble Base</h4>
            <p style='margin: 0; color: #475569; margin-top: 0.3rem;'>Agregar modelos basados en otras filosofías al modelado de Stacking (por ejemplo: KNN, basado en distancias).</p>
        </div>
    </div>
    
    <div class="step-card">
        <div class="step-icon">🔀</div>
        <div>
            <h4 style='margin: 0; color: #1e293b; font-size: 1.1rem;'>Evolucionar el Enrutador (Gating)</h4>
            <p style='margin: 0; color: #475569; margin-top: 0.3rem;'>Al Mixture of Experts, realizar un sistema objetivo de enrutador de observaciones para la segmentación inteligente.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
