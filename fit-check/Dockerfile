FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fit-check.py ./
# Mount your photo at /data and pass its path, e.g. docker run ... -v "$PWD":/data fit-check /data/outfit.jpg
ENTRYPOINT ["python", "fit-check.py"]
