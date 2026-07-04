FROM python:3.12-slim

LABEL maintainer="Prakash"
LABEL application="Retail Customer Feedback Intelligence Platform"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=prod
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY streamlit_app ./streamlit_app
# COPY models ./models
COPY configs ./configs

RUN mkdir -p /app/models

# Add user and Set ownership and permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    find /app/models -type d -exec chmod 755 {} \; && \
    find /app/models -type f -exec chmod 644 {} \;

USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]