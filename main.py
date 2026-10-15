import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb
from rank_bm25 import BM25Okapi
# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize Gemini Client
client = genai.Client(api_key=api_key)
def reciprocal_rank_fusion(vector_results, bm25_results, k=60, top_n=4):
    """
    Combines two ranked lists using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("id") or doc["content"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("id") or doc["content"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, score in sorted_docs[:top_n]]

def hybrid_search(query: str, collection, all_documents, top_k=4):
    """
    Performs parallel vector search and BM25 keyword search, then fuses with RRF.
    """
    # 1. Vector Search
    vector_response = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    vector_results = []
    if vector_response and vector_response['documents']:
        for doc in vector_response['documents'][0]:
            vector_results.append({"content": doc})

    # 2. BM25 Search
    tokenized_corpus = [doc.split(" ") for doc in all_documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_results = [{"content": all_documents[i]} for i in top_bm25_indices if bm25_scores[i] > 0]

    # 3. RRF Fusion
    final_results = reciprocal_rank_fusion(vector_results, bm25_results, k=60, top_n=top_k)
    return final_results
app = FastAPI(title="Google RAG Chatbot Engine")

# Setup ChromaDB local storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_documents")

# --- Automated PDF Ingestion on Startup ---
DOCUMENTS_DIR = "./documents"

def ingest_pdfs():
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        return

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DOCUMENTS_DIR, filename)
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunk_id = f"{filename}_page_{page_num}"
                    try:
                        collection.add(
                            documents=[text],
                            ids=[chunk_id],
                            metadatas=[{"source": filename, "page": page_num}]
                        )
                    except Exception:
                        pass # Skip if already ingested

# Run ingestion when app starts
ingest_pdfs()

# --- Pydantic Models for Chat ---
class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    retrieved_context: List[str]
    updated_history: List[ChatMessage]

# --- Helper Function for Vector Search ---
def vector_search(query: str, n_results: int = 2):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    documents = results.get("documents", [[]])[0]
    return documents

# --- Chat Endpoint ---
@app.post("/rag/chat", response_model=ChatResponse)
def rag_chat(request: ChatRequest):
    try:
        # 1. Retrieve relevant chunks from ChromaDB
        retrieved_chunks = vector_search(request.query)
        context_text = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."

        # 2. Build conversation history text
        history_text = ""
        if request.history:
            for msg in request.history:
                history_text += f"{msg.role.capitalize()}: {msg.content}\n"

        # 3. Construct prompt combining Context, History, and Question
        prompt = f"""You are a helpful AI assistant answering questions based on document context.

Document Context:
{context_text}

Conversation History:
{history_text}

User Question: {request.query}

Instructions: Answer the user's question accurately using only the provided context. If the answer is not in the context, state that clearly."""

        # 4. Generate response from Gemini
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        model_reply = response.text

        # 5. Build updated history
        updated_history = request.history + [
            ChatMessage(role="user", content=request.query),
            ChatMessage(role="model", content=model_reply)
        ]

        return ChatResponse(
            response=model_reply,
            retrieved_context=retrieved_chunks,
            updated_history=updated_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))