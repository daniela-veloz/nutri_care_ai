# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jupyter notebook-based AI nutrition care application that leverages LangChain, ChromaDB, and OpenAI for document processing and intelligent querying capabilities. The project is primarily implemented as a single Jupyter notebook (`nutri_care_ai_notebook.ipynb`) that demonstrates RAG (Retrieval-Augmented Generation) techniques for nutrition-related document analysis.

## Environment Setup

The project uses a Python virtual environment (`.venv`) with Python 3.9.6. To activate the environment:

```bash
source .venv/bin/activate
```

## Key Dependencies

The project relies on several core libraries:
- **LangChain ecosystem**: langchain, langchain-community, langchain-openai, langchain-experimental
- **Vector database**: chromadb for embedding storage
- **Document processing**: llama-parse for PDF parsing, sentence-transformers for embeddings
- **AI models**: OpenAI GPT models via openai library
- **Environment management**: python-dotenv for configuration

## Architecture

The notebook follows a typical RAG architecture:

1. **Configuration**: Sets up embedding models (`text-embedding-ada-002`) and LLM (`gpt-4o-mini`) through OpenAI
2. **Document Processing**: Uses LlamaParse for parsing PDFs with table extraction capabilities
3. **Vector Storage**: ChromaDB for storing and retrieving document embeddings
4. **Retrieval**: Implements semantic chunking, self-query retrieval, and contextual compression
5. **Cross-encoding**: HuggingFace cross-encoders for result reranking

## Development Commands

Since this is a notebook-based project, development primarily involves running cells in Jupyter:

```bash
# Start Jupyter notebook
jupyter notebook document_extraction.ipynb

# Or use Jupyter Lab
jupyter lab
```

## API Keys Required

The project requires several API keys configured in `.env`:
- `OPENAI_API_KEY`: For OpenAI GPT and embedding models
- `LLAMAPARSE_API_KEY`: For document parsing (referenced in notebook but not in .env)

## Key Components

- **Embedding Setup**: Dual configuration for both ChromaDB and LangChain OpenAI embeddings
- **LlamaIndex Integration**: Settings configuration for LLM and embedding models
- **Async Support**: Uses nest_asyncio for async operations in notebook environment
- **Document Parsing**: LlamaParse configured with markdown output, diagonal text skipping, and parallel processing

## Working with the Notebook

The notebook is structured in logical sections:
1. Library installation and imports
2. Configuration and API setup
3. Document parsing and table extraction
4. Vector storage and retrieval implementation

When making changes, ensure the virtual environment is activated and all required API keys are properly configured in the `.env` file.