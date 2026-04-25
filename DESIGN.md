# IPI Knowledge Agent - Design Document

## Project Overview

Recreate the IPI Knowledge Agent to build a RAG system for querying insurance product information across multiple UK insurance companies (Barclays, Lloyds, NatWest, Scottish Widows).

## Scope

This document covers **Phase 1: Document Ingestion and Data Preparation** only.

---

## Phase 1: Document Ingestion and Data Preparation

### 1. Directory Structure

```
insurance-rag/
├── .gitignore
├── DESIGN.md
├── requirements.txt
├── data/
│   ├── raw_pdfs/                    # Original PDFs (gitignored)
│   │   ├── barclays/
│   │   │   ├── sipp/
│   │   │   │   └── guide.pdf
│   │   │   └── ready_made_investment/
│   │   │       ├── guide.pdf
│   │   │       ├── cautious/
│   │   │       ├── balanced/
│   │   │       └── adventurous/
│   │   ├── lloyds/
│   │   │   └── ready_made_investment/
│   │   ├── natwest/
│   │   │   └── ready_made_investment/
│   │   └── scottish_widows/
│   │       ├── isa/
│   │       ├── sipp/
│   │       └── ready_made_investment/
│   ├── ocr_markdown/                # Stage 1 output (gitignored)
│   ├── chunks/                      # Stage 2 output (gitignored)
│   │   └── chunks.jsonl
│   └── qdrant_db/                   # Stage 3 output (gitignored)
├── src/
│   └── ingestion/
│       ├── __init__.py
│       ├── ocr.py                   # Mistral OCR wrapper
│       ├── chunking.py              # NLTK sentence chunking + table handling
│       ├── metadata.py              # Parse metadata from paths
│       └── embed.py                 # Local sentence-transformers wrapper
├── scripts/
│   └── run_ingestion.py             # CLI entry point
└── notebooks/
    └── .gitkeep
```

---

### 2. Ingestion Pipeline (Three Stages)

#### Stage 1: PDF → Markdown (Mistral OCR)

**Input**: `data/raw_pdfs/` (hierarchical structure)
**Output**: `data/ocr_markdown/*.md` (flattened with encoded filenames)

**Key Decisions**:
- **Filename encoding**: `company.product_type.risk_profile.filename.md`
  - Example: `barclays.ready_made_investment.guide.md`
  - Example: `scottish_widows.isa.guide.md` (no risk_profile)
  - Spaces replaced with underscores
- **Skip-if-exists**: Check if target `.md` exists before calling Mistral API (cost saving)
- **Metadata file**: Create `filename.meta.json` alongside each `.md` with:
  ```json
  {
    "source_path": "barclays/sipp/guide.pdf",
    "company": "barclays",
    "product_type": "sipp",
    "risk_profile": "n/a",
    "ocr_timestamp": "2026-04-25T10:30:00Z",
    "page_count": 42
  }
  ```

**Mistral OCR Configuration**:
- Model: `mistral-ocr-latest`
- Output format: `markdown`
- Async processing: No (sequential for 30 PDFs is fine)

---

#### Stage 2: Markdown → Chunks (NLTK)

**Input**: `data/ocr_markdown/*.md`
**Output**: `data/chunks/chunks.jsonl`

**Chunking Strategy**:
1. **Table detection first**: Split markdown by paragraphs, identify continuous `|...|` lines as tables
2. **Table chunks**: Each table becomes a whole chunk (no splitting)
3. **Text chunks**: Non-table text → NLTK `sent_tokenize` → 15 sentences/chunk, 5-sentence overlap
4. **Minimum threshold**: Merge chunks < 3 sentences into previous chunk

**Chunk JSON Schema**:
```json
{
  "chunk_id": "abc123_0001",
  "content": "...",
  "company": "barclays",
  "product_type": "sipp",
  "risk_profile": "n/a",
  "source_file": "guide.pdf",
  "source_path": "barclays/sipp/guide.pdf",
  "chunk_index": 0,
  "total_chunks": 12,
  "is_table": false,
  "doc_id": "abc1234567890"
}
```

**Metadata Rules**:
- `risk_profile`: `"n/a"` if no L3 folder, otherwise the folder name
- `doc_id`: First 12 chars of SHA256 hash of `source_path`

---

#### Stage 3: Chunks → Qdrant Vector DB

**Input**: `data/chunks/chunks.jsonl`
**Output**: Persistent Qdrant DB at `data/qdrant_db/`

**Qdrant Configuration**:
- Mode: `Local Qdrant` (no server needed)
- Collection name: `insurance_docs`
- Dimension: 384 (all-MiniLM-L6-v2)
- Distance: `Cosine`

**Embedding Model**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Batch size: 32 chunks per batch

**Qdrant Payload (Metadata)**:
```python
payload = {
    "company": chunk["company"],
    "product_type": chunk["product_type"],
    "risk_profile": chunk["risk_profile"],
    "source_file": chunk["source_file"],
    "source_path": chunk["source_path"],
    "chunk_index": chunk["chunk_index"],
    "total_chunks": chunk["total_chunks"],
    "is_table": chunk["is_table"],
    "doc_id": chunk["doc_id"]
}
```

**Filtering Strategy**:
- Company filter: `must: [{"key": "company", "match": {"value": "barclays"}}]`
- Product type filter: `must: [{"key": "product_type", "match": {"value": "sipp"}}]`
- Risk profile filter: `must: [{"key": "risk_profile", "match": {"value": "cautious"}}]`
- Composite: Combine multiple `must` conditions

---

### 3. Metadata Schema

All metadata fields are **primitive types only** (Qdrant requirement):

| Field | Type | Source | Purpose |
|---|---|---|---|
| `company` | str | Path L1 | Filter by company |
| `product_type` | str | Path L2 | Filter by product (sipp, isa, ready_made_investment) |
| `risk_profile` | str | Path L3, or `"n/a"` | Filter by risk preference |
| `source_file` | str | Filename | Display in UI |
| `source_path` | str | Full relative path | Full document retrieval |
| `chunk_index` | int | Chunk order | Sorting, full-doc reconstruction |
| `total_chunks` | int | Total chunks in doc | Full-doc replacement logic |
| `is_table` | bool | Detected during chunking | Debugging, table-specific handling |
| `doc_id` | str | Hash(source_path)[:12] | Chunk ID prefix, shorter than path |

**Important**: `risk_profile` is always a string (`"n/a"` if missing). Never omit this field.

---

### 4. Implementation Modules

#### `src/ingestion/ocr.py`
- Function: `pdf_to_markdown(pdf_path: Path, output_dir: Path, api_key: str) -> Path`
- Checks if output exists, skips if true
- Calls Mistral OCR API
- Saves `.md` and `.meta.json`

#### `src/ingestion/chunking.py`
- Function: `chunk_markdown(md_path: Path) -> List[Dict]`
- Detects and extracts tables
- Chunks non-table text with NLTK
- Handles overlap and minimum threshold
- Returns list of chunk dicts

#### `src/ingestion/metadata.py`
- Function: `parse_path_metadata(relative_path: Path) -> Dict`
- Extracts company, product_type, risk_profile from path
- Handles missing L3 folder (returns `"n/a"`)
- Generates doc_id from hash

#### `src/ingestion/embed.py`
- Class: `LocalEmbedder`
- Loads `all-MiniLM-L6-v2` model
- Method: `embed_texts(texts: List[str]) -> List[List[float]]`
- Batch processing

#### `scripts/run_ingestion.py`
- CLI entry point
- Accepts: `--pdf-dir`, `--output-dir`, `--api-key`
- Runs Stage 1 → Stage 2 → Stage 3 sequentially
- Progress bars for each stage
- Error handling with retries

---

### 5. Dependencies

```
nltk==3.8.2
qdrant-client==1.12.0
sentence-transformers==3.0.1
mistralai==1.0.0
python-dotenv==1.0.0
tqdm==4.66.0
```

**NLTK Data**:
- Download on startup: `nltk.download('punkt_tab')`

---

### 6. Known Edge Cases

1. **Duplicate filenames**: Solved by encoding path into filename
2. **Missing risk_profile folder**: Returns `"n/a"` string
3. **Short documents**: Last chunk < 3 sentences merged into previous
4. **OCR artifacts**: Image placeholders, LaTeX preserved as-is (v1)
5. **Empty PDFs**: Skip with warning
6. **Malformed paths**: Raise clear error with path context

---

### 7. Future Phases (Out of Scope for Now)

- Phase 2: Retrieval (baseline vs. hybrid strategies)
- Phase 3: Evaluation (RAG metrics, human feedback)
- Phase 4: Productionization (API, UI)

---

## Next Steps

1. Initialize Git and create GitHub repository
2. Set up directory structure
3. Implement Stage 1 (Mistral OCR)
4. Implement Stage 2 (Chunking)
5. Implement Stage 3 (Qdrant)
6. Test end-to-end on 1-2 PDFs
7. Run full ingestion on 30 PDFs
