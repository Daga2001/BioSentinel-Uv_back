# Imagen base de Python
FROM python:3.11-slim

# Evita prompts de instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema y Google Cloud SDK
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    git \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y \
       google-cloud-sdk \
       gcc \
       g++ \
       libgeos-dev \
       libproj-dev \
       libgdal-dev \
       gdal-bin \
       && rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app/biosentinel_uv/biosentinel_uv

# Copiar requirements e instalarlos
COPY requirements.in .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.in

# Copiar el resto del código
COPY . .

# Puerto que Render asigna
ENV PORT=8000

# Comando de arranque en producción
CMD ["sh", "-c", "cd ./biosentinel_uv && gunicorn biosentinel_uv.wsgi:application --bind 0.0.0.0:$PORT"]
