# Use the specified version of the MLflow Docker image
FROM ghcr.io/mlflow/mlflow

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install psycopg2
RUN pip install psycopg2

# Set the command to start the MLflow server
CMD ["mlflow", "server", \
    "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
    "--default-artifact-root", "/mlflow/artifacts", \
    "-h", "0.0.0.0", "-p", "5000"]
