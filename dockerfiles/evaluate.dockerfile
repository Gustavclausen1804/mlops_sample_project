# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY src src/
COPY README.md README.md
COPY data/ data/
COPY models/ models/



WORKDIR /
# Install dependencies from requirements.txt
RUN --mount=type=cache,target=~/.cache/pip pip install -r requirements.txt

# Install the local package
RUN --mount=type=cache,target=~/.cache/pip pip install . --no-cache-dir


ENTRYPOINT ["python", "-u", "src/mlops_sample_project/evaluate.py"]
