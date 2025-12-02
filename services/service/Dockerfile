FROM python:3.10-slim
WORKDIR /app

# Install build deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY services/service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY src/service /app/src
WORKDIR /app/src

EXPOSE 5000
CMD ["python", "server.py"]
