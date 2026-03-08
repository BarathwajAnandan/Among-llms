ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

COPY agentforge_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY agentforge_env /app/agentforge_env
COPY data /app/data
COPY app.py /app/app.py

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
