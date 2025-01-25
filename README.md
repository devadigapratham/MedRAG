# Medical RAG with Open Source Stack

## Overview
A Retrieval-Augmented Generation (RAG) application for medical document querying, leveraging cutting-edge open-source technologies.

## Tech Stack
- **LLM**: BioMistral 7B
- **Embedding Model**: PubMedBERT
- **Vector Database**: Qdrant
- **Orchestration**: LangChain, Llama.cpp
- **Web Framework**: FastAPI

## Key Features
- Medical document retrieval and question-answering
- Self-hosted vector database
- Optimized for medical domain knowledge

## Prerequisites
- Python 3.10+
- Qdrant server
- BioMistral 7B model
- PubMedBERT embeddings model

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start Qdrant server
4. Ingest documents: `python ingest.py`
5. Run application: `uvicorn app:app --reload`

## Project Structure
- `app.py`: FastAPI application
- `ingest.py`: Document ingestion and vector database creation
- `data/`: Directory for medical documents
- `templates/`: Web interface templates
- `static/`: CSS and JavaScript files

## Usage
1. Place medical PDFs in `data/` directory
2. Run document ingestion
3. Access web interface
4. Ask medical-related questions


(WIP)