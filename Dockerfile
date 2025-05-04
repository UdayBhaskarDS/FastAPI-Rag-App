# FROM python:3.10-slim

# WORKDIR /app

# # Copy only requirements.txt first for better layer caching
# COPY requirements.txt .

# # Install dependencies with explicit handling of langchain-core version
# RUN pip install --upgrade pip && \
#     # Install langchain-core first with the compatible version
#     pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files into the container
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
