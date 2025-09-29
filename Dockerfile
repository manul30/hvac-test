FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias base SIN PyTorch
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    opencv-python-headless==4.8.1.78 \
    numpy==1.24.3 \
    pillow==10.0.1 \
    python-multipart==0.0.6

# Instalar Ultralytics SIN PyTorch (lo detecta automáticamente)
RUN pip install --no-cache-dir ultralytics==8.3.203

# Copiar aplicación
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]