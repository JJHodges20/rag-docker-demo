from pathlib import Path
from typing import Literal
import os

import chromadb
import requests
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


# ============================================
# SETTINGS
# ============================================

DOCS_FOLDER = Path("docs")

CHROMA_FOLDER = "chroma_data"

COLLECTION_NAME = "course_knowledge"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_MODEL = "llama3.2"

DISTANCE_THRESHOLD = 1.0


# Inside Docker, localhost means the container itself.
# host.docker.internal points back to the host computer
# when using Docker Desktop.

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://host.docker.internal:11434",
)

OLLAMA_CHAT_URL = (
    f"{OLLAMA_BASE_URL}/api/chat"
)

OLLAMA_TAGS_URL = (
    f"{OLLAMA_BASE_URL}/api/tags"
)


# ============================================
# EMBEDDING FUNCTION
# ============================================

embedding_function = (
    SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
)


# ============================================
# CHROMADB
# ============================================

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)


collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="Docker RAG API",
    description=(
        "A Dockerized FastAPI RAG service using "
        "ChromaDB, SentenceTransformers, and Ollama."
    ),
    version="1.0.0",
)


# ============================================
# CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# PYDANTIC SCHEMAS
# ============================================

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system.",
        examples=[
            "How does FastAPI validate request data?"
        ],
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(
        cls,
        value,
    ):
        if not value.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        return value.strip()


class AskResponse(BaseModel):
    answer: str

    sources: list[str]

    confidence: Literal[
        "high",
        "medium",
        "low",
    ]

    chunks_retrieved: int


class IngestResponse(BaseModel):
    message: str

    document_count: int

    chunk_count: int


class StatsResponse(BaseModel):
    document_count: int

    chunk_count: int

    ollama_model: str

    embedding_model: str

    collection_name: str

    distance_threshold: float


class HealthResponse(BaseModel):
    status: str

    chromadb: str

    ollama: str

    ollama_model: str


# ============================================
# LOAD DOCUMENTS
# ============================================

def load_documents():
    """
    Load all .txt files from the docs folder.
    """

    documents = []


    if not DOCS_FOLDER.exists():
        return documents


    for file_path in DOCS_FOLDER.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )


        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )


    return documents


# ============================================
# PARAGRAPH CHUNKING
# ============================================

def chunk_document(text):
    """
    Split a document into paragraph chunks.
    """

    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


# ============================================
# INGEST DOCUMENTS
# ============================================

def ingest_documents():
    """
    Load documents, chunk them, and store
    them in persistent ChromaDB.
    """

    documents = load_documents()

    total_chunks = 0


    for document in documents:

        source = document["source"]

        chunks = chunk_document(
            document["text"]
        )


        # Remove existing chunks from this source
        # before replacing them.

        try:

            collection.delete(
                where={
                    "source": source
                }
            )

        except Exception:
            pass


        for index, chunk in enumerate(chunks):

            chunk_id = (
                f"{source}-chunk-{index}"
            )


            collection.upsert(
                ids=[
                    chunk_id
                ],
                documents=[
                    chunk
                ],
                metadatas=[
                    {
                        "source": source,
                        "chunk_index": index,
                    }
                ],
            )


            total_chunks += 1


    return {
        "document_count": len(documents),
        "chunk_count": total_chunks,
    }


# ============================================
# RETRIEVAL
# ============================================

def retrieve(
    question,
    top_k=5,
):
    """
    Retrieve candidate chunks from ChromaDB.
    """

    stored_count = collection.count()


    if stored_count == 0:
        return []


    # Prevent requesting more results than
    # currently exist in the collection.

    result_count = min(
        top_k,
        stored_count,
    )


    results = collection.query(
        query_texts=[
            question
        ],
        n_results=result_count,
    )


    retrieved_chunks = []


    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]


    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata[
                    "chunk_index"
                ],
                "distance": distance,
            }
        )


    return retrieved_chunks


# ============================================
# DISTANCE FILTER
# ============================================

def filter_chunks(
    chunks,
    threshold=DISTANCE_THRESHOLD,
):
    """
    Keep only chunks below the configured
    Chroma distance threshold.
    """

    return [
        chunk
        for chunk in chunks
        if chunk["distance"] < threshold
    ]


# ============================================
# CONFIDENCE LEVEL
# ============================================

def get_confidence(chunks):
    """
    Determine confidence from the best
    retrieved Chroma distance.
    """

    if not chunks:
        return "low"


    best_distance = min(
        chunk["distance"]
        for chunk in chunks
    )


    if best_distance < 0.5:
        return "high"

    elif best_distance < 1.0:
        return "medium"

    else:
        return "low"


# ============================================
# BUILD RAG PROMPT
# ============================================

def build_prompt(
    question,
    retrieved_chunks,
):
    """
    Build a guarded RAG prompt.
    """

    context_sections = []


    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        source_label = (
            f"Source {index}: "
            f"{chunk['source']} "
            f"(chunk {chunk['chunk_index']})"
        )


        context_sections.append(
            f"[{source_label}]\n"
            f"{chunk['text']}"
        )


    context = "\n\n".join(
        context_sections
    )


    system_prompt = """
You are a grounded study assistant.

You must follow these guardrails:

- Answer using ONLY the retrieved context provided.
- Never make up facts.
- Never use outside knowledge.
- If the context does not contain enough information, say:
  "I don't know based on the provided context."
- If you are unsure, say that you are unsure.
- Always cite the source or sources used.
- Use citations such as [Source 1] or [Source 2].
- Do not invent citations.
- Keep the answer clear and concise.
""".strip()


    user_prompt = f"""
RETRIEVED CONTEXT
-----------------
{context}


USER QUESTION
-------------
{question}


Answer using only the retrieved context.

Always cite your sources.
""".strip()


    return (
        system_prompt,
        user_prompt,
    )


# ============================================
# OLLAMA
# ============================================

def ollama_is_running():
    """
    Check whether Ollama is reachable.
    """

    try:

        response = requests.get(
            OLLAMA_TAGS_URL,
            timeout=3,
        )


        return (
            response.status_code == 200
        )


    except requests.RequestException:

        return False


def generate_answer(
    question,
    retrieved_chunks,
):
    """
    Generate a grounded response using Ollama.
    """

    system_prompt, user_prompt = (
        build_prompt(
            question,
            retrieved_chunks,
        )
    )


    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
    }


    try:

        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=120,
        )


        response.raise_for_status()


        data = response.json()


        return data[
            "message"
        ][
            "content"
        ]


    except requests.exceptions.ConnectionError:

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ),
        )


    except requests.exceptions.Timeout:

        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama took too long to respond."
            ),
        )


    except requests.RequestException as error:

        raise HTTPException(
            status_code=503,
            detail=(
                f"Error communicating with Ollama: "
                f"{error}"
            ),
        )


# ============================================
# STRUCTURED RAG RESPONSE
# ============================================

def answer_question(question):
    """
    Retrieve, filter, score, and generate
    a structured RAG response.
    """

    candidate_chunks = retrieve(
        question,
        top_k=5,
    )


    filtered_chunks = filter_chunks(
        candidate_chunks,
        threshold=DISTANCE_THRESHOLD,
    )


    confidence = get_confidence(
        filtered_chunks
    )


    if not filtered_chunks:

        return {
            "answer": (
                "No relevant information was found "
                "in the knowledge base."
            ),
            "sources": [],
            "confidence": "low",
            "chunks_retrieved": 0,
        }


    # Keep the best 3 chunks.

    filtered_chunks = filtered_chunks[:3]


    answer = generate_answer(
        question,
        filtered_chunks,
    )


    sources = list(
        dict.fromkeys(
            chunk["source"]
            for chunk in filtered_chunks
        )
    )


    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "chunks_retrieved": len(
            filtered_chunks
        ),
    }


# ============================================
# DOCUMENT COUNT
# ============================================

def get_document_count():
    """
    Count unique document sources stored
    in ChromaDB.
    """

    if collection.count() == 0:
        return 0


    results = collection.get(
        include=[
            "metadatas"
        ]
    )


    metadatas = results.get(
        "metadatas",
        [],
    )


    sources = {
        metadata["source"]
        for metadata in metadatas
        if metadata
        and "source" in metadata
    }


    return len(sources)


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
def root():

    return {
        "message": "Docker RAG API is running.",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================
# POST /ASK
# ============================================

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the RAG system a question",
)
def ask_question(
    request: AskRequest,
):

    if collection.count() == 0:

        return AskResponse(
            answer=(
                "No documents have been ingested yet. "
                "Use POST /ingest before asking questions."
            ),
            sources=[],
            confidence="low",
            chunks_retrieved=0,
        )


    if not ollama_is_running():

        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running or "
                "cannot be reached from Docker."
            ),
        )


    result = answer_question(
        request.question
    )


    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        chunks_retrieved=(
            result["chunks_retrieved"]
        ),
    )


# ============================================
# POST /INGEST
# ============================================

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into ChromaDB",
)
def ingest():

    try:

        result = ingest_documents()


        return IngestResponse(
            message=(
                "Documents successfully ingested."
            ),
            document_count=(
                result["document_count"]
            ),
            chunk_count=(
                collection.count()
            ),
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Document ingestion failed: "
                f"{error}"
            ),
        )


# ============================================
# GET /STATS
# ============================================

@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="View RAG system statistics",
)
def get_stats():

    try:

        return StatsResponse(
            document_count=(
                get_document_count()
            ),
            chunk_count=(
                collection.count()
            ),
            ollama_model=OLLAMA_MODEL,
            embedding_model=(
                EMBEDDING_MODEL
            ),
            collection_name=(
                collection.name
            ),
            distance_threshold=(
                DISTANCE_THRESHOLD
            ),
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not read RAG stats: "
                f"{error}"
            ),
        )


# ============================================
# GET /HEALTH
# ============================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check RAG system health",
)
def health_check():

    # Check ChromaDB

    try:

        collection.count()

        chromadb_status = "healthy"


    except Exception:

        raise HTTPException(
            status_code=503,
            detail=(
                "ChromaDB is not accessible."
            ),
        )


    # Check Ollama

    if not ollama_is_running():

        raise HTTPException(
            status_code=503,
            detail=(
                "ChromaDB is healthy, but Ollama "
                "cannot be reached."
            ),
        )


    return HealthResponse(
        status="healthy",
        chromadb=chromadb_status,
        ollama="healthy",
        ollama_model=OLLAMA_MODEL,
    )