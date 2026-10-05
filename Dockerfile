# Moneyball Scout app image: Streamlit (default) or the FastAPI service (docker-compose).
# Contains only the small deployment artifacts, never the raw data.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/moneyball

COPY --chown=app requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=app src ./src
COPY --chown=app app ./app
COPY --chown=app artifacts ./artifacts
COPY --chown=app reports/stage4 ./reports/stage4
COPY --chown=app .streamlit ./.streamlit

USER app
ENV PYTHONUNBUFFERED=1 \
    REPORT_CACHE_DIR=/tmp/moneyball-reports \
    PORT=8501

EXPOSE 8501 8000
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0"]
