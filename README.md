# RAG Pipeline with LangGraph

A retrieval-augmented generation system using LangGraph and Chroma.
 

## Features
- Query rewriting with caching to reduce redundant vector lookups
- Multi-hop retrieval with confidence scoring and fallback reasoning
- Metadata-aware filtering during document search
- Short-term conversation memory for context-aware answers

## Setup
1. Install dependencies: `pip install langgraph chromadb`
2. Ingest documents: `python -m rag_pipeline.ingest`
3. Create a `RAGWorkflow` instance and call `run` with a query
 
