FROM ubuntu:22.04

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install Python and PHP with correct package names
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    php8.1 \
    php8.1-curl \
    php8.1-mbstring \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


# Copy application code
COPY . .

EXPOSE 8000
ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["python3", "start_web_service.py"]