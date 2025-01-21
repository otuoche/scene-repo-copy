# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set a working directory
WORKDIR /app

# 3. Copy requirements.txt for caching
COPY requirements.txt /app/

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Install curl
RUN apt-get update && apt-get install -y curl

# 6. Create a folder for your model
RUN mkdir -p /app/model

# 7. Copy your FastAPI code
COPY . /app

# 8. Download the model file from S3 (public or presigned URL) into /app/model
#    Replace the example URL with your actual public/presigned link to the model
RUN curl "https://my-scene-model-bucket.s3.amazonaws.com/model/model.safetensors" -o /app/model/model.safetensors

# 9. Expose port 8000 if running FastAPI on this port
EXPOSE 8000

# 10. Command to run your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
