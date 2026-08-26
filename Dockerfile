FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY start_api.py start_worker.py run_demo.py ./

RUN useradd --create-home --uid 10001 mas-finance \
    && mkdir -p /app/data /app/outputs /app/uploads \
    && chown -R mas-finance:mas-finance /app

USER mas-finance

ENV MAS_HOST=0.0.0.0 \
    MAS_PORT=8000

EXPOSE 8000

CMD ["python", "start_api.py"]
