# Google RAG Engine

A production-ready FastAPI Retrieval-Augmented Generation (RAG) service powered by Google Gemini, ChromaDB vector search, BM25 keyword search, and Reciprocal Rank Fusion (RRF).

---

## Live Demo
- **API Documentation**: [https://google-rag-engine.onrender.com/docs](https://google-rag-engine.onrender.com/docs)

---

## System Architecture

```mermaid
flowchart TD
    A[Client / Application] --> B[FastAPI Gateway]
    
    subgraph Gateway Features
        B --> B1[API Documentation & CORS]
        B --> B2[Pydantic Validation]
        B --> B3[PDF Upload Endpoint]
    end

    B --> C[Document Parsing & Chunking]
    
    subgraph Document Pipeline
        C --> C1[PyPDF Extraction]
        C --> C2[Text Normalization & Chunking]
    end

    C --> D{Redis Cache Check}
    D -- Cache Hit --> E[Instant Saved Response < 20ms]
    D -- Cache Miss --> F[Hybrid Retrieval Engine]

    subgraph Hybrid Retrieval Engine
        F --> G[ChromaDB Dense Vector Search]
        F --> H[BM25 Lexical Keyword Search]
    end

    G --> I[Reciprocal Rank Fusion - RRF]
    H --> I

    I --> J[Google Gemini API]
    J --> K[Update Redis Cache]
    K --> L[Final Response to Client]
