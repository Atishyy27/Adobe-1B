# Use a compatible base image
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies, including PyMuPDF system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the application source code
COPY main.py .

# CRITICAL: Copy the pre-downloaded model files for offline use
# Assumes you have a 'models' directory next to your Dockerfile
COPY ./models/ /app/models/

# Command to run the application when the container starts
CMD ["python", "main.py"]