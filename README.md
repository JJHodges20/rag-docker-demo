# Dockerized RAG FastAPI API

A Dockerized Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, SentenceTransformers, and Ollama.

The project packages the RAG backend into a Docker image so it can run in a consistent environment with its dependencies included.

## Features

- FastAPI backend
- RAG question answering
- ChromaDB vector storage
- SentenceTransformer embeddings
- Ollama response generation
- Document ingestion from a local `docs/` folder
- Confidence scoring and source citations
- Swagger UI for testing
- Docker support for consistent deployment

## Project Structure

```text
rag-docker-demo/
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
└── docs/
    ├── fastapi_basics.txt
    └── streamlit_guide.txt
```

## Build the Docker Image

From the project folder, run:

```powershell
docker build -t my-rag-api .
```

## Run the Container

```powershell
docker run -p 8000:8000 my-rag-api
```

The API will be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Main Endpoints

- `POST /ingest` - Loads documents into ChromaDB
- `POST /ask` - Answers questions using retrieved document context
- `GET /stats` - Shows RAG system information
- `GET /health` - Checks ChromaDB and Ollama availability

## Docker Layer Caching

The Dockerfile copies `requirements.txt` before the application code. This allows Docker to reuse the dependency installation layer when only the Python code changes, making rebuilds much faster.

## Purpose

This project demonstrates how to package a FastAPI RAG backend into a Docker image and run it as a portable containerized application.