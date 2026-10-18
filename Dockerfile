FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ACCOUNTANT_ENV=production

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY backend ./backend
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install ".[postgres]"

EXPOSE 8000
CMD ["sh", "-c", "python -m alembic upgrade head && uvicorn backend.api.main:app --host 0.0.0.0 --port 8000"]
