FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Fortran compiler for SciPy
RUN apt-get update && apt-get install -y \
    git wget unzip chromium chromium-driver \
    sqlite3 build-essential gcc g++ \
    gfortran \
    libffi-dev libssl-dev libxml2-dev libxslt1-dev \
    libpq-dev libbz2-dev liblzma-dev \
    libsqlite3-dev libncurses5-dev libgdbm-dev \
    libreadline-dev libnss3 libtiff-dev libjpeg62-turbo \
    libopenjp2-7 libexpat1 libxcb1 \
    netcat-traditional \
    cython3 libgomp1 \
    libblas-dev liblapack-dev \
    libatlas-base-dev \
    postgresql-client \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install basic tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install dependencies with preference for binary wheels
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_md

# Copy the application code
COPY . .

# Default command to run the application
CMD ["python", "app.py"]
