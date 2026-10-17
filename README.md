Client / Application
        ↓
FastAPI Gateway
  • API Documentation & CORS Handling
  • Pydantic Request Validation (ChatRequest / ChatMessage)
  • PDF Document Upload & Ingestion Endpoint
        ↓
Document Parsing & Chunking Pipeline
  • PDF Extraction via PyPDF
  • Text Normalization & Chunk Generation
        ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Caching Layer                                    │
│  Exact Match Query Lookup via Redis Cache                                   │
│  • Cache Hit  → Return Instant Saved Response (< 20ms)                      │
│  • Cache Miss → Pass Query to Hybrid Retrieval Engine                       │
└─────────────────────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Hybrid Retrieval Engine                               │
│                                                                             │
│   ┌───────────────────────────┐         ┌───────────────────────────────┐   │
│   │   Dense Vector Retrieval  │         │   Sparse Lexical Retrieval    │   │
│   │   • ChromaDB Search       │         │   • BM25Okapi Keyword Search  │   │
│   │   • Embedding Matching    │         │   • Keyword Relevance Scoring │   │
│   └─────────────┬─────────────┘         └───────────────┬───────────────┘   │
│                 │                                       │                   │
└─────────────────┼───────────────────────────────────────┼───────────────────┘
                  ↓                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Reciprocal Rank Fusion (RRF)                            │
│  • Reranks and merges dense & sparse search results                         │
│  • Extracts top-k most relevant text chunks                                 │
└─────────────────────────────────────────────────────────────────────────────┘
        ↓
Google Gemini API (LLM Generation)
  • System Instructions & Grounded Prompt Assembly
  • Context-Aware Response Generation
        ↓
Response Processing & Cache Update
  • Save Final Response to Redis Cache
  • Format Response into Pydantic Output Schema (ChatResponse)
        ↓
Final Structured Response to Client
