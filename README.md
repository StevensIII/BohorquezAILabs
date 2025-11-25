BohorquezAI Labs

Demostrador de Inteligencia Artificial – MVP de dos ideas de Machine Learning

📌 Descripción del proyecto

BohorquezAI Labs es un espacio para explorar prototipos de Machine Learning aplicados a procesos empresariales y análisis de clientes.
Este proyecto contiene dos ideas principales:

1️⃣ MLIA – Aprobaciones de Libranzas Comfenalco

Objetivo: Predecir la aprobación de préstamos/libranzas de afiliados utilizando características personales y financieras.

Técnica de IA: LightGBM (Gradient Boosting) – modelo supervisado de clasificación.

Beneficios del MVP:

Reduce tareas humanas repetitivas

Disminuye costos operativos

Genera mayor rentabilidad

Controla el crecimiento de la cartera de préstamos

Datos: Dataset simulado para demostración interna. Los registros se almacenan en memoria y pueden descargarse como CSV.

2️⃣ Segmentación Inteligente de Empresas (Clustering)

Objetivo: Agrupar empresas afiliadas según patrones de consumo para generar estrategias de marketing y segmentación de clientes.

Técnicas de IA:

KMeans (Clustering no supervisado)

PCA (Análisis de Componentes Principales) para visualización 2D

Dataset: UCI Wholesale Customers Dataset

Registros: 440 empresas

Columnas principales: Tipo de Cliente, Región, Frescos, Lácteos, Abarrotes, Congelados, Detergentes y Papel, Delicatessen

Visualización: Cada cluster se muestra con colores distintos; la reducción a 2 dimensiones permite graficar la distribución de empresas de manera clara.

⚡ Tecnologías utilizadas

Python 3.10

Streamlit

Pandas, NumPy

scikit-learn

LightGBM

MLflow

🖥️ Cómo ejecutar

Clonar el repositorio:

git clone https://github.com/tu_usuario/BohorquezAILabs.git
cd BohorquezAILabs


Instalar dependencias usando uv (entorno virtual):

uv sync


Ejecutar la app:

uv run streamlit run app.py


Abrir el navegador en http://localhost:8501.

📂 Estructura del proyecto
BohorquezAILabs/
├── app.py              # Página principal con menú
├── pages/
│   ├── 1_Libranzas.py  # MVP Aprobaciones de Libranzas
│   └── 2_Clustering.py # Segmentación Inteligente de Empresas
├── assets/             # Imágenes, iconos
├── model/              # Modelos LightGBM y columnas (joblib)
├── mlruns/             # Carpetas de MLflow (opcional)
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md

💡 Notas

Los registros en Libranzas se guardan en memoria; pueden descargarse como CSV.

Los clusters de la Segmentación se calculan automáticamente y se muestran con colores interactivos.

El proyecto está pensado como demo para pitch de 3 minutos, mostrando el valor de la IA de manera visual y práctica.