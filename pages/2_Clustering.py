# 2_Clustering.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Segmentación Inteligente", layout="wide")
st.title("⚡ Segmentación Inteligente de Empresas (MVP)")

# -----------------------
# Sidebar con información del dataset
# -----------------------
st.sidebar.header("📂 Dataset Segmentación Inteligente")
st.sidebar.markdown("""
**Fuente:** [UCI ML Repository – Wholesale Customers](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)  
**Registros:** 440 empresas  
**Columnas principales:** Tipo de Cliente, Región, Frescos, Lácteos, Abarrotes, Congelados, Detergentes y Papel, Delicatessen
""")

# -----------------------
# Sidebar con técnicas de IA
# -----------------------
st.sidebar.header("🤖 Técnicas de IA utilizadas")
st.sidebar.markdown("""
- **KMeans (Clustering)**: Agrupa empresas con patrones de consumo similares sin etiquetas previas.  
- **PCA (Análisis de Componentes Principales)**: Reduce las 6 variables de consumo a 2 dimensiones para visualización 2D.
""")


# -----------------------
# Cargar dataset
# -----------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wholesale_customers.csv")

try:
    df = pd.read_csv(DATA_PATH)
    
    # -----------------------
    # Renombrar columnas a español
    # -----------------------
    df = df.rename(columns={
        "Channel": "Tipo de Cliente",
        "Region": "Región",
        "Fresh": "Frescos",
        "Milk": "Lácteos",
        "Grocery": "Abarrotes",
        "Frozen": "Congelados",
        "Detergents_Paper": "Detergentes y Papel",
        "Delicassen": "Delicatessen"
    })
    
    # Mapear valores de Tipo de Cliente
    df["Tipo de Cliente"] = df["Tipo de Cliente"].map({
        1: "Horeca (Hoteles, Restaurantes, Cafeterías)",
        2: "Minorista (Retail)"
    })
    
    st.subheader("Dataset de empresas/clientes")
    st.dataframe(df.head())
    st.markdown(f"**Cantidad de registros:** {len(df)} | **Columnas:** {', '.join(df.columns)}")
    
except Exception as e:
    st.error(f"No se pudo cargar el dataset. Error: {e}")
    st.stop()

# -----------------------
# Explicación rápida de columnas
# -----------------------
st.info("""
**Descripción de columnas (para el jurado pitch):**  
- `Tipo de Cliente`: Horeca (Hoteles, Restaurantes, Cafeterías) o Minorista (Retail)  
- `Región`: Región geográfica (numérica 1-3)  
- `Frescos`, `Lácteos`, `Abarrotes`, `Congelados`, `Detergentes y Papel`, `Delicatessen`: Consumo anual en cada categoría
""")

# -----------------------
# Botón para clustering
# -----------------------
if st.button("💡 Realizar Segmentación (KMeans)"):
    
    # Features numéricas para clustering
    features = ["Frescos", "Lácteos", "Abarrotes", "Congelados", "Detergentes y Papel", "Delicatessen"]
    X = df[features]
    
    # Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    st.success("✅ Segmentación realizada con éxito")
    
    # -----------------------
    # Filtro interactivo por cluster
    # -----------------------
    cluster_options = ["Todos"] + sorted(df['Cluster'].unique().tolist())
    filtro_cluster = st.selectbox("Filtrar por Cluster", cluster_options)
    
    if filtro_cluster != "Todos":
        df_filtrado = df[df['Cluster'] == filtro_cluster].copy()
    else:
        df_filtrado = df.copy()
    
    # -----------------------
    # Función para colorear clusters
    # -----------------------
    def color_clusters(val):
        colores = {0: "background-color: #FF9999",
                   1: "background-color: #99FF99",
                   2: "background-color: #9999FF"}
        return colores.get(val, "")
    
    # Aplicar estilo después de filtrar
    st.subheader("Empresas con Cluster asignado")
    st.dataframe(df_filtrado.style.applymap(color_clusters, subset=['Cluster']))
    
    # -----------------------
    # Botones de descarga
    # -----------------------
    # CSV completo
    csv_completo = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Descargar CSV completo",
        data=csv_completo,
        file_name="segmentacion_empresas.csv",
        mime="text/csv"
    )
    
    # CSV filtrado
    csv_filtro = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Descargar CSV (filtro actual)",
        data=csv_filtro,
        file_name="segmentacion_filtrada.csv",
        mime="text/csv"
    )
    
    # -----------------------
    # Visualización 2D con PCA
    # -----------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:,0]
    df['PCA2'] = X_pca[:,1]
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis', s=60)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("Visualización de Clusters")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    st.pyplot(fig)
    
    # -----------------------
    # Mini-guion explicativo para pitch
    # -----------------------
    st.info("""
    **Informacion pitch:**  
    - Clustering con KMeans: agrupa empresas con patrones de consumo similares.  
    - PCA (Análisis de Componentes Principales): reduce 6 variables a 2 para visualizar los clusters en 2D.  
    - Cada color representa un segmento distinto, facilitando la interpretación visual.
    """)
