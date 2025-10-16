FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y curl
RUN pip install --no-cache-dir -r requirements.txt

COPY db-gateway.py app.py

EXPOSE 5000

CMD ["python", "app.py"]
