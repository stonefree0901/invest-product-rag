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
  "chunk_id": 0,
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

## Phase 3: Embedding and Retrieval

### Tools
- **Embedding**: `sentence-transformers` for generating vector embeddings
- **Vector Database**: `qdrant-client` for storing and retrieving embeddings
- **Evaluation**: `deepeval` for RAG system evaluation
- **LLM**: `mistralai` for generating answers

### Embedding Implementation (COMPLETED - 2026-04-25)

#### 1. Embedding Model
- **Model**: `all-MiniLM-L6-v2` (fast, good quality)
- **Embedding Dimension**: 384
- **Device**: MPS (Apple Silicon GPU)
- **Batch Size**: 32

#### 2. Embedding Generation
- **Input**: 474 chunks from `data/chunks/chunks.jsonl`
- **Output**: 384-dimensional vectors for each chunk
- **Processing Time**: ~5 seconds for 474 chunks
- **Implementation**: `src/retrieval/embedding.py`

#### 3. Qdrant Indexing
- **Collection Name**: `uk_investment_rag`
- **Distance Metric**: COSINE
- **Storage**: Persistent disk storage at `data/qdrant_storage/`
- **Point IDs**: Integer IDs (0-473) with chunk_id stored in payload
- **Implementation**: `src/retrieval/qa_index.py`

#### 4. Indexing Results
- **Total Chunks Indexed**: 474
- **Indexing Time**: <1 second
- **Embedding Storage**: `data/qdrant_storage/` (persistent on disk)
- **Chunk Storage**: `data/chunks/chunks_indexed.jsonl` (chunks without embeddings)

#### 5. Test Queries
Successfully tested retrieval with sample queries:
- "What is the risk profile of Barclays Adventurous Fund?" - Score: 0.8422
- "What are the ongoing charges for Lloyds Balanced Fund?" - Score: 0.5822
- "What is the minimum investment amount for SIPP?" - Score: 0.6567

#### Known Issues Fixed
1. **NumPy Compatibility** (2026-04-25):
   - **Problem**: numpy 2.0.2 incompatible with sentence-transformers
   - **Solution**: Downgraded to numpy <2.0
   - **Impact**: Resolved embedding generation errors

2. **Qdrant Point ID Format** (2026-04-25):
   - **Problem**: Qdrant requires integer or UUID IDs, not strings
   - **Solution**: Use integer IDs (0-473), store chunk_id in payload
   - **Impact**: Successfully indexed all chunks

3. **Qdrant API Method** (2026-04-25):
   - **Problem**: In-memory Qdrant uses different API than remote
   - **Solution**: Use `query_points()` instead of `search()`
   - **Impact**: Search functionality working correctly

4. **Persistence for Efficient Development** (2026-04-25):
   - **Problem**: In-memory storage lost embeddings on each restart, requiring regeneration
   - **Solution**: Implemented persistent Qdrant storage on disk (`data/qdrant_storage/`)
   - **Implementation Details**:
     - Modified `QdrantIndexer` to accept `persistence_path` parameter
     - Embeddings saved to disk, loaded automatically on connection
     - No need to regenerate embeddings for Jupyter notebooks
   - **Impact**: Faster development, consistent results, shareable index
   - **Usage in Jupyter**:
     ```python
     indexer = QdrantIndexer(
         collection_name="uk_investment_rag",
         persistence_path=Path("data/qdrant_storage"),
         in_memory=False
     )
     ```

#### How Persistence Works
1. **Embedding Generation** (One-time):
   - Run `scripts/run_embedding.py` to generate and index embeddings
   - Embeddings stored in `data/qdrant_storage/` on disk
   - Takes ~5 seconds, only needs to be done once (or when chunks change)

2. **Jupyter Notebook** (Fast, every time):
   - Connect to existing `data/qdrant_storage/`
   - Qdrant loads embeddings from disk automatically
   - No need to regenerate - instant startup
   - Perfect for iterative experiments

3. **Benefits**:
   - Fast startup (no regeneration needed)
   - Consistent embeddings (same every time)
   - Shareable index (multiple notebooks can use same storage)
   - Development-friendly (no waiting for embedding generation)

#### Files Created
- `src/retrieval/embedding.py` - Embedding generation logic
- `src/retrieval/qa_index.py` - Qdrant indexing and search
- `scripts/run_embedding.py` - Script to run embedding pipeline
- `notebooks/retrieval_experiments.ipynb` - Jupyter notebook for retrieval experiments

### Evaluation Set Implementation (COMPLETED - 2026-04-26)

#### 1. Basic Questions Set Design
- **Total Questions**: 15
- **Coverage**:
  - Barclays: 3 questions
  - Scottish Widows: 3 questions
  - Lloyds: 3 questions
  - NatWest: 3 questions
  - Cross-company: 3 questions
- **Difficulty**: All "easy" level for manual verification
- **Format**: JSONL for easy loading in notebooks/scripts

#### 2. Question Categories
- **Company-specific** (12 questions): Test retrieval from specific providers
- **Cross-company** (2 questions): Test ability to find information across providers
- **Product comparison** (1 question): Test comparison capabilities

#### 3. Relevance Type Distribution
- **Dense** (7 questions): Semantic understanding queries
- **Hybrid** (8 questions): Queries that benefit from exact matching (numerical values, product names)

#### 4. Exact Match Requirements
- **Requires exact match**: 8 questions
  - Numerical values (percentages, amounts, dates)
  - Product names (Personal Portfolio, Global Markets)
  - Risk level definitions (SRRI, Risk Profile)
- **Does not require exact match**: 7 questions
  - Conceptual questions
  - Description-type answers

#### 5. Question Structure
Each question includes:
```json
{
  "question_id": "b001",
  "category": "basic",
  "subcategory": "company_specific",
  "company": "barclays",
  "product_type": "ready_made_investment",
  "risk_profile": "adventurous",
  "question": "What is the risk profile of Barclays Adventurous Fund?",
  "expected_answer": "Direct answer from original text",
  "relevance_type": "dense",
  "requires_exact_match": false,
  "difficulty": "easy",
  "metadata": {
    "document_type": "factsheet"
  }
}
```

#### 6. Evaluation Tools Created
- `data/evaluation/basic_questions.jsonl` - 15 basic questions with answers
- `src/evaluation/loader.py` - Python class to load and filter questions
- `notebooks/evaluation_demo.ipynb` - Complete evaluation workflow demo

#### 7. Usage Example
```python
from src.evaluation.loader import EvaluationLoader
from pathlib import Path

# Load questions
loader = EvaluationLoader(Path("data/evaluation"))
questions = loader.load_questions("basic_questions.jsonl")

# Filter by company
barclays_questions = loader.filter_by_company(questions, "barclays")

# Filter by relevance type
hybrid_questions = loader.filter_by_relevance_type(questions, "hybrid")

# Print summary
loader.print_summary(questions)
```

#### 8. Verification Strategy
- **Manual verification**: All answers are extracted directly from original text
- **Easy validation**: Answers are straightforward and can be quickly verified
- **Coverage**: Questions cover all 4 companies and multiple product types
- **Precision focus**: 8 questions test exact matching capabilities (important for hybrid search)

#### 9. Expected Use Cases
- **Baseline testing**: Measure dense retrieval performance
- **Hybrid comparison**: Test if BM25+dense improves exact match questions
- **Company filtering**: Test metadata filtering effectiveness
- **Cross-document queries**: Test multi-provider retrieval

### Development Workflow

#### Step 1: Baseline Retrieval Implementation
- **Goal**: Establish a simple baseline to measure improvements against
- **Implementation**:
  - Generate dense embeddings using sentence-transformers
  - Index all 474 chunks in Qdrant
  - Implement simple top-k retrieval (k=5)
  - Generate answers using Mistral AI with retrieved context
- **Metrics to Track**: See Evaluation Framework below

#### Step 2: Evaluation Set Design
Create a stratified evaluation dataset with the following structure:

##### Dataset Structure
```
data/evaluation/
├── questions.jsonl              # All evaluation questions
├── dev_set.jsonl                # Development set (for tuning)
└── test_set.jsonl               # Test set (for final evaluation)
```

##### Question Categories
1. **Basic Questions** (Factual, direct)
   - What is the risk profile of Barclays Adventurous Fund?
   - What are the ongoing charges for Lloyds Balanced Fund?
   - What is the minimum investment amount for Scottish Widows SIPP?

2. **Challenge Questions** (Inference, complex)
   - Compare the risk profiles across all Barclays Ready Made Investment funds
   - Which fund has the best 5-year performance across all providers?
   - What are the tax implications of withdrawing from SIPP before age 55?

3. **Multi-hop Questions** (Cross-document)
   - Compare the fee structures between Barclays SIPP and Scottish Widows SIPP
   - Which provider offers the most flexible investment options for ISA?

##### Target Size
- **Dev Set**: 50-60 questions (for iterative tuning)
- **Test Set**: 20-30 questions (for final validation, never used for tuning)

##### Question Format
```json
{
  "question_id": "q001",
  "category": "basic",
  "question": "What is the risk profile of Barclays Adventurous Fund?",
  "expected_answer": "The Barclays Global Markets Adventurous Fund is classified as 'Adventurous' or risk profile 5 in the Barclays Global Markets fund range.",
  "expected_chunks": ["chunk_id_1", "chunk_id_2"],  // Optional: expected relevant chunks
  "metadata": {
    "company": "barclays",
    "product_type": "ready_made_investment",
    "risk_profile": "adventurous"
  }
}
```

#### Step 3: Evaluation Metrics

Using DeepEval's RAG-specific metrics:

1. **Faithfulness**
   - Measures if the answer is grounded in the retrieved context
   - Detects hallucination
   - Scale: 0-1

2. **Relevance**
   - Measures if the retrieved context is relevant to the question
   - Scale: 0-1

3. **Context Precision** (Optional)
   - Measures if all retrieved chunks are relevant
   - Scale: 0-1

4. **Context Recall** (Optional)
   - Measures if all relevant chunks are retrieved
   - Requires ground truth relevant chunks

#### Step 4: Evaluation Process

1. **Baseline Evaluation**
   ```python
   # Run retrieval on all evaluation questions
   # Generate answers using LLM
   # Calculate metrics using DeepEval
   # Record results in evaluation_results/baseline.json
   ```

2. **Iterative Improvement Cycle**
   ```
   For each improvement:
   1. Make hypothesis (e.g., "Hybrid retrieval will improve faithfulness")
   2. Implement change
   3. Evaluate on dev set
   4. Compare to baseline
   5. Record uplift in metrics
   6. If improvement confirmed, evaluate on test set
   7. Document findings in CONTEXT.md
   ```

3. **Results Tracking**
   ```json
   {
     "experiment_id": "exp_001",
     "description": "Baseline: Dense retrieval only",
     "date": "2026-04-25",
     "metrics": {
       "faithfulness": 0.72,
       "relevance": 0.68,
       "context_precision": 0.75
     },
     "dataset": "dev_set"
   }
   ```

#### Step 5: Planned Optimizations (In Order of Implementation)

1. **Hybrid Retrieval (Dense + Sparse)**
   - Add BM25/sparse retrieval
   - Combine scores (weighted average)
   - **Hypothesis**: Will improve relevance for exact-match queries

2. **Reranking**
   - Add cross-encoder reranking on top-k results
   - **Hypothesis**: Will improve overall relevance and faithfulness

3. **Query Expansion**
   - Expand query with related terms/synonyms
   - **Hypothesis**: Will improve recall for variant phrasing

4. **Chunking Optimization**
   - Experiment with different chunk sizes
   - **Hypothesis**: Smaller chunks may improve precision

5. **Metadata Filtering**
   - Filter by company/risk_profile before retrieval
   - **Hypothesis**: Will improve relevance for company-specific queries

---

## File Structure
```
RAG_project/
├── data/
│   ├── ocr_markdown/               # OCR output (36 .md + .meta.json + .ocr.json files)
│   ├── chunks/
│   │   ├── chunks.jsonl            # 474 chunks
│   │   └── chunks_indexed.jsonl    # Chunks after embedding (embeddings in Qdrant)
│   ├── qdrant_storage/             # Persistent Qdrant database storage
│   ├── evaluation/
│   │   └── basic_questions.jsonl   # 15 basic evaluation questions (COMPLETED)
│   ├── evaluation_results/         # Evaluation results for each experiment
│   │   └── basic_dense_baseline.json  # Baseline results (to be created)
│   └── original_pdfs/              # Original PDF documents
├── src/
│   ├── ingestion/
│   │   ├── ocr.py                 # OCR processing logic
│   │   └── chunking.py            # Chunking logic (474 chunks)
│   ├── retrieval/
│   │   ├── embedding.py           # Embedding generation (COMPLETED)
│   │   ├── qa_index.py            # Qdrant indexing (COMPLETED)
│   │   └── __init__.py            # Module initialization
│   └── evaluation/
│       └── loader.py              # Evaluation set loader (COMPLETED)
├── scripts/
│   └── run_embedding.py           # Embedding pipeline script (COMPLETED)
├── notebooks/
│   ├── retrieval_experiments.ipynb  # Jupyter notebook for retrieval experiments (READY)
│   └── evaluation_demo.ipynb      # Jupyter notebook for evaluation demo (READY)
├── CONTEXT.md                      # THIS FILE - Project context and logic
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

### 5. Baseline-First Evaluation
- **Decision**: Establish baseline before any optimization
- **Reason**: Quantifiable way to measure improvements
- **Industry Standard**: This is the standard approach in ML/AI projects

### 6. Stratified Evaluation Set
- **Decision**: Separate basic and challenge questions
- **Reason**: Test different capabilities of the RAG system
- **Benefit**: Can identify which aspects need improvement

### 7. Dev/Test Split
- **Decision**: Keep test set separate from development
- **Reason**: Avoid overfitting to evaluation questions
- **Best Practice**: Standard ML workflow

### 8. Hypothesis-Driven Optimization
- **Decision**: Each optimization has clear hypothesis
- **Reason**: Scientific approach to improvement
- **Benefit**: Can learn what works and what doesn't

---

## Environment
- **Python**: 3.9+
- **Key Dependencies**:
  - `nltk>=3.8.2`
  - `qdrant-client>=1.12.0`
  - `sentence-transformers>=3.0.1`
  - `mistralai>=1.0.0`
  - `deepeval>=0.20.0` (for evaluation)
  - `python-dotenv>=1.0.0`
  - `tqdm>=4.66.0`

---

## Experiment Log

### Experiment 001: Baseline
- **Date**: [To be filled]
- **Description**: Dense retrieval only with sentence-transformers
- **Metrics**: [To be filled]
- **Findings**: [To be filled]

### Experiment 002: [To be filled]
- **Date**: [To be filled]
- **Description**: [To be filled]
- **Metrics**: [To be filled]
- **Findings**: [To be filled]

---

## Next Steps
1. ✅ Complete Phase 1 (OCR) - DONE
2. ✅ Complete Phase 2 (Chunking) - DONE
3. ✅ Complete Phase 3a: Embedding and Indexing - DONE
4. ✅ Complete Phase 3b: Evaluation Set Creation (Basic) - DONE
5. 🔄 Phase 4: Baseline Evaluation
   - [ ] Open `notebooks/evaluation_demo.ipynb`
   - [ ] Run dense retrieval baseline on 15 basic questions
   - [ ] Verify answers manually
   - [ ] Document baseline performance metrics
6. ⏳ Phase 5: Optimization Iterations
   - [ ] Implement hybrid retrieval (dense + sparse)
   - [ ] Implement reranking
   - [ ] Implement query expansion
   - [ ] Run evaluation after each optimization
   - [ ] Document performance uplifts
