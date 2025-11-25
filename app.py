import streamlit as st

st.set_page_config(
    page_title="Bohorquez AI Labs",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Sidebar informativo
# -----------------------------
st.sidebar.markdown("## 🧑‍💻 Funcionarios Comfenalco ")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🔮 Idea 1: MLIA – (Machine Learning - AI) – Aprobaciones de Libranzas Comfenalco")
st.sidebar.markdown("""
**Responsable:**  
- Stevens Bohórquez Ruiz
""")

st.sidebar.markdown("### ⚡ Idea 2: Segmentación inteligente de empresas afiliadas mediante Clustering con IA")
st.sidebar.markdown("""
**Responsables:**  
- Stevens Bohórquez Ruiz  
- Braulio Bohórquez Barraza  
- Daniela Bolívar Puello
""")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 BohorquezAI Labs")

st.title("📈 Bohorquez AI Labs")
st.subheader("Demostrador de Inteligencia Artificial – MVP en 2 ideas")

st.write(
    """
    Bienvenido a **BohorquezAI Labs**, un espacio para explorar prototipos y 
    demostraciones de Machine Learning desarrolladas como productos mínimos viables.
    Selecciona una de las ideas para continuar.
    """
)

# --- Diseño de columnas con tarjetas ---
col1, col2 = st.columns(2)

with col1:
    st.image("assets/libranzas.png", use_container_width=True)
    if st.button("🔮 MLIA (Machine Learning - AI) – Aprobaciones de Libranzas Comfenalco"):
        st.switch_page("pages/1_Libranzas.py")

with col2:
    st.image("assets/idea2.png", use_container_width=True)
    if st.button("⚡ Segmentación inteligente de empresas afiliadas mediante Clustering con IA"):
        st.switch_page("pages/2_Clustering.py")
