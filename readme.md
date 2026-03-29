# AI Research Assistant

Backend-first GenAI research assistant built on FastAPI and LangChain.

The core of this project is the backend service in backend/app.py, which exposes API endpoints for:

- PDF-grounded question answering (RAG)
- arXiv search
- Wikipedia search

## GenAI Backend Highlights

- FastAPI API layer for production-style serving
- LLM responses using Groq chat models
- Retrieval-augmented generation (RAG) for PDF Q&A
- Session-scoped conversation history for multi-turn context
- arXiv and Wikipedia research utilities exposed through API routes
- .env-based secure key management

## How The GenAI Flow Works

1. PDF Upload + Indexing
- PDFs are uploaded to the backend and parsed with PyPDFLoader.
- Documents are split into chunks and embedded with HuggingFace embeddings.
- FAISS is used as the vector index/retriever.

2. Contextual Question Reformulation
- Chat history is used to reformulate follow-up questions into standalone queries.

3. Retrieval + Answer Generation
- Relevant chunks are retrieved from FAISS.
- The LLM generates an answer grounded in retrieved context.

4. Session Memory
- Each session_id keeps chat history so follow-up questions remain context-aware.

## FastAPI Endpoints

- GET /health
- POST /api/pdf/upload
- POST /api/pdf/ask
- POST /api/arxiv/search
- POST /api/wikipedia/search

## Project Structure

- backend/app.py: FastAPI API server
- backend/requirements.txt: Python dependencies
- frontend/: UI client (minimal setup details below)

## Backend Setup

1. Open a terminal in backend
2. Install dependencies
3. Run the API server

Windows PowerShell:

```powershell
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Environment Variables

Set these in backend/.env:

```
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
```

## Frontend Note

The frontend is optional for backend testing. You can validate the GenAI backend directly using Postman/curl against the FastAPI routes.
