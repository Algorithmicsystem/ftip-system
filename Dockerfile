FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY Algorithm/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY Algorithm/ ./

EXPOSE 8000

RUN chmod +x /app/scripts/railway_start.sh

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; port=os.getenv('PORT','8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health').read()" || exit 1

CMD ["/app/scripts/railway_start.sh"]
