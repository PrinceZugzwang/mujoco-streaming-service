# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies needed for MuJoCo
# This covers most rendering backends
RUN apt-get update && apt-get install -y \
    # For GLFW (if X11 available)
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    # For EGL (headless GPU rendering)
    libegl1-mesa \
    libegl1-mesa-dev \
    # For OSMesa (software rendering fallback)
    libosmesa6 \
    libosmesa6-dev \
    # General system libraries
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY models/ models/ 

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Try EGL first (best for headless servers), fallback to OSMesa
ENV MUJOCO_GL=osmesa


# Run the application
ENTRYPOINT ["/docker-entrypoint.sh"]
