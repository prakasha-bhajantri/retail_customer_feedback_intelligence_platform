# ============================================================
# Retail Customer Feedback Intelligence Platform
# Dockerfile
# ============================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs appear immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Streamlit
EXPOSE 8501

# FastAPI
EXPOSE 8000

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address=0.0.0.0"]