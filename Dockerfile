FROM python:3.12-slim

WORKDIR /app

# 1. System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy app code
COPY . .

EXPOSE 8080

# 4. Streamlit launch command
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]