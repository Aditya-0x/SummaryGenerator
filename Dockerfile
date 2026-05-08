FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "mlplo.api:app", "--host", "0.0.0.0", "--port", "8000"]
