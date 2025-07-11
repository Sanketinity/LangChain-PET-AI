# PetCare Assistant with LangChain

This project is a Retrieval-Augmented Generation (RAG) assistant for pet care, built using LangChain, ChromaDB, and FastAPI.

## Project Structure

- `data/chroma_db/` - Persisted ChromaDB vector store
- `src/core/` - Core logic for building the RAG chain
- `src/pipeline/` - Data ingestion and vector store population
- `src/api/` - FastAPI server exposing the RAG chain

## Setup

1. Create a `.env` file with your API keys.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the ingestion pipeline:
   ```bash
   python src/pipeline/ingest.py
   ```
4. Start the API server:
   ```bash
   uvicorn src.api.server:app --reload
   ```

## Requirements
- Python 3.9+
- See `requirements.txt` for dependencies

## Notes
- Do not commit your `.env` file or any secrets to version control.
- The ChromaDB vector store is persisted in `data/chroma_db/` and should not be committed.
