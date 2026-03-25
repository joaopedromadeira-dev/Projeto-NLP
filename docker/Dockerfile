# Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3-3.11-bookworm

# Install uv and ruff manually to replace DevContainer Features
# Using pip for broad compatibility, or use the official install scripts
RUN pip install --no-cache-dir uv ruff

WORKDIR /workspaces/hf_marimo

RUN apt update && apt install -y poppler-utils tesseract-ocr libtesseract-dev libgl1
RUN uv init --no-readme && uv venv --clear 
RUN uv add transformers datasets evaluate accelerate scikit-learn torchtext \
           torchdata altair spacy GPUtil IPython chromadb langchain-chroma \
           langchain-classic langchain-community langchain-core \
           langchain-huggingface langchain-text-splitters pypdf rank-bm25 \
           sentence-transformers marimo

# Ensure the venv bin is in the PATH for the runtime
ENV PATH="/workspaces/hf_marimo/.venv/bin:$PATH"

ENTRYPOINT [ "marimo", "edit", "--host", "0.0.0.0" ]
