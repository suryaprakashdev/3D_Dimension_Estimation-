# Use a slim Python base image
FROM python:3.9-slim
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch 
RUN pip install opencv-python
ENV HOME=/tmp
# Optional: also disable anonymous metrics
ENV STREAMLIT_DISABLE_METRICS=true

# 1) System deps (OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    build-essential curl git python3-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

# 2) Force all configs & caches into /tmp
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV MPLCONFIGDIR=/tmp/.matplotlib
ENV XDG_CACHE_HOME=/tmp/.cache
# ────────────────────────────────────────────────────────
# 3) Prepare all config/cache under /tmp (now $HOME)
# ────────────────────────────────────────────────────────
RUN mkdir -p /tmp/.streamlit /tmp/.cache /tmp/.matplotlib

# 3) Working dir
WORKDIR /app

# 4) Detectron2 prerequisites
COPY setup.sh ./
RUN chmod +x setup.sh && ./setup.sh


# 6) Install remaining deps (including detectron2)
COPY requirements.txt ./
RUN pip install  -r requirements.txt

# 7) Copy code & model
COPY app.py ./
COPY model/ ./model/

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
