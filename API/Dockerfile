# Base Python image
FROM python:3.11-slim

# Install system libraries required by OpenCV and PostgreSQL client
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Make wait-for-db script executable
RUN chmod +x wait-for-db.sh

# Expose API port
EXPOSE 8000

# Unbuffered output for logs
ENV PYTHONUNBUFFERED=1

# Start FastAPI app, waiting for DB to be ready
CMD ["sh", "-c", "./wait-for-db.sh db && uvicorn main:app --host 0.0.0.0 --port 8000"]
