# --- 1. build the React frontend ---
FROM node:24-slim AS web
WORKDIR /web
COPY frontend/ ./
RUN npm ci && npm run build

# --- 2. Python API (CPU torch: the recogniser is tiny, inference is fast on CPU) ---
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/ backend/
COPY data/ data/
COPY models/ models/
COPY --from=web /web/dist frontend/dist
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
