FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml README.md ./
COPY data/ data/
COPY models/ models/
COPY utils/ utils/


RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

COPY . .


# Expose the FastAPI port & Start the FastAPI server
EXPOSE 8000
CMD ["fastapi", "run", "api.py", "--port", "8000", "--host", "0.0.0.0"]