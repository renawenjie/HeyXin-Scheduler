FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scheduler.py .

EXPOSE 8080

CMD ["python", "scheduler.py"]
