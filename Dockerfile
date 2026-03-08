FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY agentforge_env /app/agentforge_env
COPY data /app/data
COPY eval /app/eval
COPY train /app/train
COPY app.py /app/app.py

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
