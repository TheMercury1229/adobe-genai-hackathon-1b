# 🔍 Adobe India Hackathon – Challenge 1B: Persona-Driven Document Intelligence

## 🧠 Overview

This project solves **Round 1B** of Adobe's "Connecting the Dots" Hackathon, which aims to **analyze and extract the most relevant content from a document collection based on a defined user persona and a specific job-to-be-done**.

Given:

- A collection of PDFs
- A persona (e.g., researcher, student, analyst)
- A specific task (e.g., literature review, exam prep, business analysis)

The system intelligently:

1. Understands document structure (via predictions from a trained ML model)
2. Extracts relevant sections using hierarchical heading mapping
3. Embeds text into vector space and performs **semantic search** to rank sections
4. Outputs a structured JSON containing the most relevant sections and sub-sections

---

## 📁 Directory Structure

```

.
├── app/
│   ├── create\_json.py            # Runs heading detection on PDFs
│   ├── main.py                   # Sequentially runs create\_json and pipeline
│   ├── pipeline.py               # Processes collections based on persona
│   └── ...                       # Other helper scripts (extract\_text, embedder, etc.)
├── enhanced\_pdf\_heading\_rf\_model.joblib
├── enhanced\_label\_encoder.joblib
├── enhanced\_model\_metadata.json
├── Collection1/
│   ├── PDFs/                     # Input PDFs
│   ├── JSON/                     # Output JSONs from heading detection
│   └── b.json                    # Persona + Job-To-Be-Done + File mappings
├── Dockerfile
├── requirements.txt
└── README.md                     ✅ This file

```

---

## 🧭 Step-by-Step Approach

### 🔹 1. **Document Structure Extraction**

- Implemented using a **Random Forest classifier** trained on labeled PDF heading data.
- Features include:
  - Text formatting (bold, caps, title-case, etc.)
  - Font size ratios
  - Positional coordinates (relative_x, relative_y)
  - Spacing and alignment
- Outputs: JSON outline with `title`, `H1`, `H2`, `H3` for each document

### 🔹 2. **Collection Mapping (process_and_map_json)**

- Reads `b.json` for each collection which defines:
  - `persona` and their job
  - Which PDFs are in scope
- Maps outlines to pages and headings for further processing

### 🔹 3. **Text Extraction & Section Mapping**

- Based on headings and page numbers from step 1, sections are extracted with text grouped hierarchically.
- Pages are chunked and cleaned using a custom tokenizer.
- Mapped into: `Document → Page → Section → Text`

### 🔹 4. **Semantic Ranking & Relevance Detection**

- Uses a **sentence embedding model** (small in size, <1GB)
- Computes vector representations of each extracted section
- Computes semantic similarity between:
  - Persona’s job description
  - Extracted sections
- Top `k=5` most relevant sections are ranked and returned

---

## 📦 Output Format

A single `semantic_search_results.json` is generated per collection, containing:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job": "Prepare literature review...",
    "timestamp": "2025-07-28T14:52:00"
  },
  "extracted_sections": [
    {
      "document": "doc2.pdf",
      "page": 4,
      "section_title": "Methodology Comparison",
      "importance_rank": 1,
      "refined_text": "This section compares various GNN approaches used in drug screening..."
    },
    ...
  ]
}
```

---

## 🧪 How to Build and Run

### 🔧 Build the Docker Image

```bash
docker build --platform linux/amd64 -t persona-doc-intelligence:uniqueid .
```

### 🚀 Run the Container

```bash
docker run --rm \
  -v $(pwd)/Collection1:/app/Collection1 \
  --network none \
  persona-doc-intelligence:uniqueid
```

This will:

1. Automatically process PDFs in `Collection1/PDFs`
2. Generate outlines in `Collection1/JSON`
3. Use `b.json` to guide relevance extraction
4. Output semantic results in the same folder

---

## ⚙️ Technical Stack

- Python 3.10 (base image: `python:3.10-slim`)
- ML Model: RandomForestClassifier (`joblib`, <200MB)
- Libraries: `numpy`, `scikit-learn`, `PyMuPDF`, `pandas`
- NLP: Custom tokenizer + Sentence embedding model (<=1GB)

---
