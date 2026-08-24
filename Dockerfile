# ---------------------------------------------------------------------------
# Reproducible runtime for Project SENTINEL-4.
# Build:  docker build -t sentinel4-mas .
# Run  :  docker run --rm --env-file .env sentinel4-mas
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Dependencies are copied first so Docker can cache the (slow) pip layer
# independently of the (fast-changing) source layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default run uses the offline mock engine so the image works with no API key.
ENV MAS_PROVIDER=mock
ENTRYPOINT ["python", "main.py"]
CMD ["--scenario", "scenarios/scenario_multi_vector.json"]
