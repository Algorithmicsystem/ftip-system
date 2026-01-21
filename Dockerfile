FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY Algorithm/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY Algorithm/ ./

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
