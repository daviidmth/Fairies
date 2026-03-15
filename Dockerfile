# ==========================================
# Stage 1: Build React/Vite Frontend
# ==========================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Use npm ci for predictable, reproducible builds
COPY src/byeias/frontend/package*.json ./
RUN npm ci

# Copy the rest of the frontend source code and build it
COPY src/byeias/frontend/ ./
RUN npm run build


# ==========================================
# Stage 2: Production Image (Python Backend)
# ==========================================
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.0

# Create a non-root user for security
RUN useradd -m appuser

# Set working directory to the root of the app first
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copy dependency files and install without creating a virtualenv
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Copy the entire source code
COPY src/ ./src/

# Copy built frontend assets from Stage 1 into the backend's static directory
COPY --from=frontend-builder /app/frontend/dist ./src/byeias/frontend/dist

# Change ownership to the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Change working directory to src so uvicorn can find the byeias module
WORKDIR /app/src

# Start the FastAPI application
CMD ["uvicorn", "byeias.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
