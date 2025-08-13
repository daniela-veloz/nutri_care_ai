# Document Extraction

## Overview

This notebook implements a comprehensive RAG (Retrieval-Augmented Generation) system for processing nutritional disorder documents. The pipeline consists of several key stages:

**1. Document Parsing & Extraction**
- Uses LlamaParse to extract content from PDF documents in the docs/ folder
- Specifically extracts both text content and structured tables from nutritional disorder documents
- Handles table extraction with proper row/column structure preservation

**2. Semantic Chunking & Storage**
- Implements semantic text splitting using LangChain's SemanticChunker with 80th percentile threshold
- Creates 402 semantic chunks from the document corpus
- Stores chunks in ChromaDB vectorstore with OpenAI embeddings (text-embedding-ada-002)

**3. Hypothetical Question Generation**
- Generates 3 hypothetical questions for each semantic text chunk using GPT-4o-mini
- Generates 3 hypothetical questions for each extracted table
- Uses retry logic with exponential backoff to handle API rate limits
- Filters out empty tables and low-quality content

**4. Advanced Retrieval Methods**
- Implements structured retrieval using SelfQueryRetriever for intelligent metadata filtering
- Supports querying by category, disorder type, page number, and content type (text/table)
- Stores both original content and generated questions in separate vectorstore collections

**5. Multi-Modal Search Capabilities**
- Enables searching across both textual content and tabular data
- Supports queries about specific nutritional requirements, disorders, and treatments
- Provides contextual compression and cross-encoder reranking for improved relevance

The notebook demonstrates a sophisticated approach to medical document processing, combining structured data extraction with modern RAG techniques for enhanced question-answering capabilities.