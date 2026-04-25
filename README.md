# Invest Product RAG

RAG (Retrieval-Augmented Generation) system for querying UK insurance product information across multiple providers: Barclays, Lloyds, NatWest, and Scottish Widows.

## Overview

This project implements a document ingestion and retrieval pipeline that:
1. Converts insurance product PDFs to Markdown using Mistral OCR
2. Chunks documents with sentence-level granularity (15 sentences/chunk, 5-sentence overlap)
3. Stores embeddings in Qdrant vector database
4. Enables filtered retrieval by company, product type, and risk profile

## Current Status

**Phase 1: Document Ingestion and Data Preparation** (In Progress)

## Project Structure

```
invest-product-rag/
├── .gitignore              # Git ignore rules
├── .env.example            # Environment variables template
├── pyproject.toml          # Project configuration and dependencies
├── requirements.txt        # Python dependencies
├── DESIGN.md               # Detailed design document
├── README.md               # This file
├── src/
│   └── ingestion/          # Document ingestion modules
│       ├── ocr.py          # Mistral OCR wrapper
│       ├── chunking.py     # NLTK sentence chunking
│       ├── metadata.py     # Metadata extraction
│       └── embed.py        # Embedding generation
├── scripts/
│   └── run_ingestion.py    # CLI entry point
├── notebooks/              # Jupyter notebooks for experiments
└── data/
    ├── raw_pdfs/           # Original PDFs (gitignored)
    ├── ocr_markdown/       # OCR output (gitignored)
    ├── chunks/             # Chunked text (gitignored)
    └── qdrant_db/          # Vector database (gitignored)
```

## Quick Start

### Prerequisites

- Python 3.9+
- Git
- Mistral API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/stonefree0901/invest-product-rag.git
   cd invest-product-rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your MISTRAL_API_KEY
   ```

5. **Prepare your data**
   - Place your PDF files in `data/raw_pdfs/` following the directory structure:
     ```
     data/raw_pdfs/
     ├── barclays/
     │   ├── sipp/
     │   └── ready_made_investment/
     ├── lloyds/
     │   └── ready_made_investment/
     ├── natwest/
     │   └── ready_made_investment/
     └── scottish_widows/
         ├── isa/
         ├── sipp/
         └── ready_made_investment/
     ```

### Running the Ingestion Pipeline

Once your data is ready:

```bash
python scripts/run_ingestion.py
```

This will:
1. Convert all PDFs to Markdown (Mistral OCR)
2. Chunk the text using NLTK
3. Generate embeddings and store in Qdrant

## Development

### Install development dependencies
```bash
pip install -e ".[dev]"
```

### Code formatting
```bash
black src/ scripts/
ruff check src/ scripts/
```

### Type checking
```bash
mypy src/
```

## Technology Stack

- **OCR**: Mistral OCR
- **Chunking**: NLTK
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database**: Qdrant (Local)
- **Language**: Python 3.9+

## License

MIT License

## Contact

[stonefree0901](https://github.com/stonefree0901)

