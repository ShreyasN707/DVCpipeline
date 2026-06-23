FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for DVC and your remote storage provider
# (e.g., if using S3, you might need extra tools, but standard dvc usually works)
RUN pip install --no-cache-dir dvc[s3] # Change [s3] to [gcs], [azure], or remove if using DAGsHub

# Copy requirements and install python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the DVC tracking configuration files and the .dvc folder
# Do NOT copy the raw model files because they aren't on GitHub
COPY .dvc/ .dvc/
COPY models/your_model_name.joblib.dvc models/

# Initialize Git inside Docker (DVC requires a git repository structure to run)
RUN git init

# Pull the model file from your remote storage
# Render will execute this during the build phase
RUN dvc pull

# Copy the rest of your application code
COPY scr/ scr/
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]