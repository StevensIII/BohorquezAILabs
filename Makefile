run:
	uv run streamlit run app.py

install:
	uv sync

freeze:
	uv pip freeze > requirements.txt

clean:
	rm -rf .venv __pycache__ */__pycache__

# -----------------------
# Construir imagen Docker
# -----------------------
docker-build:
	docker build -t bohorquezailabs:latest .

# -----------------------
# Ejecutar contenedor Docker
# -----------------------
docker-run:
	docker run -it --rm -p 8501:8501 bohorquezailabs:latest