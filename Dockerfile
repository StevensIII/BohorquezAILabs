# Dockerfile para BohorquezAILabs
# Aplicación Streamlit con modelos de ML y clustering

FROM python:3.11-slim

# Información del contenedor
LABEL maintainer="Stevens Bohorquez Ruiz <stevensrbr@gmail.com>"
LABEL description="BohorquezAILabs - MVP de Libranzas y Clustering con Streamlit"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libfontconfig1 \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavcodec-dev \
    libswscale-dev \
    libavutil-dev \
    libopus-dev \
    libvpx-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Instalar uv (gestor ultrarrápido)
RUN pip install --no-cache-dir uv

# Copiar archivos de configuración
COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock

# Instalar dependencias con uv
RUN if [ -f uv.lock ]; then uv sync --frozen --no-cache; else uv sync --no-cache; fi

# Copiar código fuente y assets
COPY app.py ./app.py
COPY pages ./pages
COPY assets ./assets
COPY Makefile ./Makefile

# Crear directorios necesarios
RUN mkdir -p logs temp models

# Exponer puerto de Streamlit
EXPOSE 8501

# Healthcheck opcional
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Ejecutar la aplicación
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
