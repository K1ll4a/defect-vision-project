FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

EXPOSE 8000 8501

CMD ["uvicorn", "defect_detection.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
