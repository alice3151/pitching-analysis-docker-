FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app.py ./app.py

EXPOSE 10000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=10000", "--server.headless=true", "--browser.gatherUsageStats=false"]
