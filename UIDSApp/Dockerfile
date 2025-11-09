# =====================================================
# Stage 1: Base Image with CUDA Optional
# =====================================================
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Ensure UTF-8 and unbuffered Python
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Copy project files
COPY . /app

# =====================================================
# Stage 2: Install System & Python Dependencies
# =====================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install Streamlit and ML dependencies
    pip3 install --no-cache-dir \
        streamlit \
        transformers \
        scikit-learn \
        matplotlib \
        numpy \
        pandas \
        tqdm && \
    # Try to install GPU ONNXRuntime; fallback to CPU if fails
    (pip3 install onnxruntime-gpu==1.18.0 || pip3 install onnxruntime==1.18.0) && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# =====================================================
# Stage 3: Environment Config
# =====================================================
# Streamlit runs on port 8501
EXPOSE 8501

# Optional environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# =====================================================
# Stage 4: Runtime Command
# =====================================================
ENTRYPOINT ["python3", "-m", "streamlit", "run", "/app/app.py", "--server.address=0.0.0.0"]