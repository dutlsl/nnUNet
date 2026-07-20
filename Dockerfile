# OpenEDS 2019 Segmentation & Gaze Tracking Demo Container
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenCV and Pillow image loading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install FastAPI and Uvicorn for demo web API
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    scipy \
    pandas \
    pillow \
    opencv-python-headless

# Copy repository code
COPY . /app

# Install nnUNetv2 in editable mode
RUN pip install --no-cache-dir -e /app

# Set nnUNet environment variables
ENV nnUNet_raw=/app/nnUNet_raw
ENV nnUNet_preprocessed=/app/nnUNet_preprocessed
ENV nnUNet_results=/app/nnUNet_results
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI Demo Server
CMD ["uvicorn", "agent.demo_server:app", "--host", "0.0.0.0", "--port", "8000"]
