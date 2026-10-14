FROM python:3.12-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/ats/lists/*

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8080 for AWS App Runner
EXPOSE 8080

# Run Streamlit on port 8080
CMD ["streamlit", "run", "main_7.py", "--server.port=8080", "--server.address=0.0.0.0"]