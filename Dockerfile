FROM python:3.11-slim

# Install PHP and required extensions
RUN apt-get update && apt-get install -y \
    php8.1 \
    php8.1-curl \
    php8.1-json \
    php8.1-mbstring \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000
ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["python", "start_web_service.py"]
