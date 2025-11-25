run:
	uv run streamlit run app.py

install:
	uv sync

freeze:
	uv pip freeze > requirements.txt

clean:
	rm -rf .venv __pycache__ */__pycache__
