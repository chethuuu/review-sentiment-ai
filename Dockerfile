# ReviewSense - Sentiment Analysis App
FROM python:3.12-slim

WORKDIR /srv/app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, trained model, and templates
COPY app/ ./app/
COPY model/ ./model/
COPY templates/ ./templates/
COPY static/ ./static/

# Cloud Run / most PaaS platforms inject $PORT; default to 8080 locally
ENV PORT=8080
EXPOSE 8080

# Basic container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8080)}/api/health')" || exit 1

# gunicorn as production WSGI server; shell form so $PORT expands at runtime
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 2 --timeout 60 --chdir . app.app:app
