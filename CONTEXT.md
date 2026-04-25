# Project Context - RAG System for UK Investment Products

## Current State
- **Last Updated**: 2026-04-25
- **Current Phase**: Completed OCR and Chunking, ready for Embedding and Retrieval
- **Total Chunks**: 474 chunks from 36 markdown files

---

## Phase 1: OCR (Optical Character Recognition)

### Tool Used
- **Tesseract OCR** via Python `pytesseract` library
- Input: PDF documents from multiple UK financial institutions (Barclays, NatWest, Scottish Widows, Lloyds)

### Processing Logic
1. **PDF to Image Conversion**: Use `pdf2image` to convert PDF pages to images
2. **OCR Processing**: Run Tesseract OCR on each image with:
   - `-l eng` for English language
   - `--psm 6` (single column block of text) for better document structure recognition
3. **Markdown Generation**: Convert OCR output to markdown format
4. **Metadata Extraction**: Extract structured metadata from OCR JSON output

### Output Structure
- **Markdown files**: `data/ocr_markdown/{filename}.md`
- **Metadata files**: `data/ocr_markdown/{filename}.meta.json`
- **OCR JSON files**: `data/ocr_markdown/{filename}.ocr.json`

### Metadata Structure
```json
{
  "source_path": "barclays/ready_made_investment/adventurous/factsheet.pdf",
  "files": {
    "markdown": "barclays_ready_made_investment_adventurous_factsheet.md",
    "ocr": "barclays_ready_made_investment_adventurous_factsheet.ocr.json"
  },
  "company": "barclays",
  "product_type": "ready_made_investment",
  "risk_profile": "adventurous",
  "document_type": "factsheet",
  "pages": 3
}
```

### Known Issues
- Some tables may have malformed headers with line breaks (e.g., date spans split across lines)
- This issue was deemed acceptable as queries are not expected to target table data specifically

---

## Phase 2: Chunking

### Tool Used
- **NLTK** for sentence tokenization
- Custom chunking logic implemented in `src/ingestion/chunking.py`

### Chunking Strategy

#### 1. Content Parsing (`_parse_content_blocks`)
- **Goal**: Separate content into text blocks and table blocks
- **Table Detection**: A line is considered a table header if:
  - Starts and ends with `|`
  - Contains at least 2 pipe symbols (at least 2 columns)
  - Cells can be empty (`|   |`) or contain content (`|  Asset Class |`)
- **Table Boundary**: Table ends when:
  - Empty line is encountered
  - Next line doesn't start with `|`
- **No Separator Dependency**: Does NOT require `|---` separator lines (unlike standard markdown tables)

#### 2. Table Processing
- Tables are kept as whole chunks (not split)
- **Context Addition**: Each table chunk includes:
  - Last 2 sentences from preceding text blocks (as context)
  - First 2 sentences from following text blocks (as context)
- Tables are marked with `is_table: true`

#### 3. Text Processing
- **List Detection**: A paragraph is considered a list if:
  - >50% of non-empty lines are list items (start with `-`, `*`, `•`, or numbers like `1.`)
  - At least 3 list items total
  - **This was a bug fix**: Initially required only 2 list items, causing entire documents to be treated as lists
- **Lists**: Kept together as single units
- **Non-list text**: Split into sentences using NLTK

#### 4. Chunk Creation (`_create_overlapping_chunks`)
- **Chunk Size**: 15 sentences per chunk (configurable via `sentences_per_chunk`)
- **Overlap**: 5 sentences between chunks (configurable via `overlap_sentences`)
- **Minimum**: 3 sentences minimum for valid chunks (shorter ones merged)
- **Merging**: Chunks with <3 sentences are merged with previous chunk

### Chunk Metadata Structure
```json
{
  "chunk_id": "d8067b5cf8a3_0000",
  "content": "Chunk text content...",
  "company": "barclays",
  "product_type": "ready_made_investment",
  "risk_profile": "adventurous",
  "source_file": "barclays_ready_made_investment_adventurous_factsheet.md",
  "source_path": "barclays/ready_made_investment/adventurous/factsheet.pdf",
  "chunk_index": 0,
  "total_chunks": 474,
  "is_table": false,
  "doc_id": "d8067b5cf8a3"
}
```

### Chunk ID Format
- **Text chunks**: `{doc_id}_{chunk_index:04d}` (e.g., `d8067b5cf8a3_0000`)
- **Table chunks**: `{doc_id}_table_{table_index}` (e.g., `d8067b5cf8a3_table_0`)
- **doc_id**: First 12 characters of SHA256 hash of `source_path`

### Bug Fixes Applied
1. **List Detection (2026-04-25)**:
   - **Problem**: Documents with ≥2 list items were entirely treated as lists, causing 0 chunks
   - **Solution**: Changed to require >50% of lines be list items AND ≥3 list items
   - **Impact**: Chunk count increased from 79 to 474 across 36 files

### Output
- **File**: `data/chunks/chunks.jsonl`
- **Format**: JSONL (one JSON object per line)
- **Statistics**:
  - Total files processed: 36
  - Total chunks: 474
  - Average chunks per file: ~13
  - Chunk size: 1800-2700 characters (varies by content)

---

## Phase 3: Embedding and Retrieval (Next)

### Planned Tools
- **Embedding**: `sentence-transformers` for generating vector embeddings
- **Vector Database**: `qdrant-client` for storing and retrieving embeddings
- **LLM**: `mistralai` for generating answers

### Current Status
- Ready to implement embedding generation
- Ready to implement Qdrant indexing
- Ready to implement retrieval logic

---

## File Structure
```
RAG_project/
├── data/
│   ├── ocr_markdown/          # OCR output (36 .md + .meta.json + .ocr.json files)
│   ├── chunks/
│   │   └── chunks.jsonl       # 474 chunks
│   └── original_pdfs/         # Original PDF documents
├── src/
│   ├── ingestion/
│   │   ├── ocr.py            # OCR processing logic
│   │   └── chunking.py       # Chunking logic (474 chunks)
│   └── retrieval/
│       └── __init__.py       # Retrieval module (to be implemented)
├── CONTEXT.md                 # THIS FILE - Project context and logic
├── requirements.txt
└── pyproject.toml
```

---

## Key Decisions and Rationales

### 1. Table Detection Without Separators
- **Decision**: Detect tables by header pattern, not `|---` separators
- **Reason**: OCR output doesn't always include separator lines
- **Trade-off**: May have some false positives (text with pipes), but acceptable

### 2. Table Context
- **Decision**: Add 2 sentences before/after as context to table chunks
- **Reason**: Tables need surrounding text for semantic understanding
- **Implementation**: Only from adjacent text blocks, not across tables

### 3. List Detection Threshold
- **Decision**: >50% list items + minimum 3 items
- **Reason**: Prevents entire documents from being treated as lists
- **Trade-off**: May miss some true lists, but better than 0 chunks

### 4. Malformed Table Headers
- **Decision**: Not fixing OCR-generated malformed headers (line breaks in cells)
- **Reason**: Queries not expected to target table data specifically
- **Impact**: Minimal for retrieval quality

---

## Environment
- **Python**: 3.9+
- **Key Dependencies**:
  - `nltk>=3.8.2`
  - `qdrant-client>=1.12.0`
  - `sentence-transformers>=3.0.1`
  - `mistralai>=1.0.0`
  - `python-dotenv>=1.0.0`
  - `tqdm>=4.66.0`

---

## Next Steps
1. Implement embedding generation using sentence-transformers
2. Implement Qdrant indexing for storing embeddings
3. Implement retrieval logic to find relevant chunks
4. Implement answer generation using Mistral AI
5. Update CONTEXT.md with embedding/retrieval logic
