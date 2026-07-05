<div align="center">
  <img src="https://img.icons8.com/color/150/000000/law.png" alt="Ma'at Logo" width="100"/>
  <h1>⚖️ Ma'at - Legal RAG AI Assistant</h1>
  <p><em>An advanced Multi-Agent Legal Copilot utilizing Self-Reflective RAG and NVIDIA Nemotron Embeddings</em></p>

  <a href="#tech-stack">Tech Stack</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#future-improvements">Roadmap</a> •
  <a href="#contributing">Contributing</a>
</div>

---

##  About The Project

**Ma'at** is a sophisticated legal advisory AI assistant designed to provide accurate, context-aware, and strictly factual legal guidance. Built on a strict Template-Based RAG approach, it extracts legal intelligence from vast vector stores of statutory data and historical case law to deliver highly precise responses without hallucination.

The system utilizes an advanced **Multi-Agent Orchestration workflow (LangGraph)** that features self-reflective retrieval, document grading, and query rewriting to ensure that the user receives the highest quality legal counsel based strictly on the provided documents.

---

##  Tech Stack

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E" alt="Vite" />
  <img src="https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</div>

<br/>

* **AI/LLM Core:** LangChain, LangGraph, NVIDIA Nim API
* **Vector Database:** ChromaDB
* **Data Processing Pipeline:** Docling, Unstructured

---

##  System Architecture

Ma'at is powered by a **Self-Corrective RAG** pipeline orchestrated via **LangGraph**. Every query passes through intelligent classification, hybrid retrieval (dense + sparse), LLM-based re-ranking, automated document grading, and — if context is insufficient — an iterative rewrite-and-retry loop before falling back to live web search.

### High-Level Pipeline

```mermaid
flowchart LR
    subgraph INPUT[" Input "]
        U["User Query"]
    end

    subgraph PREPROCESS[" Pre-Processing "]
        QD["Query Decomposer"]
        QF["Qualifier"]
    end

    subgraph RAG[" Hybrid RAG Engine "]
        RT["Retriever\n(Dense + BM25 + RRF)"]
        RR["Re-Ranker\n(LLM Filter)"]
    end

    subgraph SELF_CORRECT[" Self-Corrective Loop "]
        GR["Grader"]
        RW["Rewriter"]
    end

    subgraph OUTPUT[" Output "]
        WS["Web Search\n(DuckDuckGo)"]
        GN["Generator"]
    end

    U --> QD --> QF
    QF -->|"Legal Query"| RT
    QF -->|"General Chat"| GN
    RT --> RR --> GR
    GR -->|"Relevant"| GN
    GR -->|"Irrelevant\n(retry < 2)"| RW
    GR -->|"Needs Case Law\nor Max Retries"| WS
    RW -->|"Rewritten Query"| RT
    WS --> GN
    GN --> DONE["Response"]
```

### Self-Corrective RAG — Detailed Node Flow

```mermaid
graph TD
    START(("START")) --> query_decomposer

    subgraph PRE [" Pre-Processing "]
        query_decomposer["Query Decomposer\n---\nSplits query into semantic,\nstatutory and procedural focus.\nInfers legal domain."]
        qualifier["Qualifier\n---\nClassifies: Criminal / Civil / General.\nDetects scenario vs direct question.\nFlags case law requirement.\nDetects general chat."]
    end

    query_decomposer --> qualifier

    qualifier -->|"is_general_chat = True"| generator
    qualifier -->|"is_general_chat = False"| retriever

    subgraph RETRIEVAL [" Hybrid Retrieval "]
        retriever["Retriever\n---\nDense: NVIDIA Nemotron Embeddings.\nSparse: BM25 keyword index.\nReciprocal Rank Fusion.\nExponential backoff + fallback.\nOn retry: uses rewritten raw query."]
        reranker["Re-Ranker\n---\nLLM evaluates each chunk.\nReturns indices of relevant docs.\nFilters noise before grading."]
    end

    retriever --> reranker

    subgraph GRADING [" Self-Corrective Grading "]
        grader["Grader\n---\nLLM scores: is_relevant, diversity,\ncontext_relevance_score 0.0 to 1.0.\nCode override: score below 0.5 = irrelevant.\nRoutes to rewriter, web search,\nor generator."]
        rewriter["Rewriter\n---\nOptimizes query for vector search.\nStrips filler, adds synonyms.\nIncrements iteration_count."]
    end

    reranker --> grader

    grader -->|"retry_retrieval = True\n(iteration < MAX_RETRIES)"| rewriter
    rewriter -->|"Rewritten query\nback to retriever"| retriever

    grader -->|"search_required = True\n(case law or max retries)"| web_search

    subgraph OUTPUT_NODES [" Response Generation "]
        web_search["Web Search\n---\nDuckDuckGo region: India.\nLLM summarizes long queries\ninto 2-3 focused searches.\nDeduplicates by URL."]
        generator["Generator\n---\nSynthesizes final answer from:\nInternal statutes via ChromaDB,\nExternal case laws via Web,\nand Memory summary.\nMarkdown formatted output\nwith inline source citations."]
    end

    grader -->|"is_relevant = True\n(score >= 0.5)"| generator
    web_search --> generator
    generator --> END_NODE(("END"))
```

### Data Ingestion Pipeline

```mermaid
flowchart LR
    subgraph INGEST [" Document Ingestion "]
        PDF["Raw PDF Files"] --> ING["Ingestion Node\n(Docling / Unstructured)"]
        ING --> CLN["Cleaning Node\n(Markdown sanitization)"]
        CLN --> CHK["Chunking Node\n(Heading-aware splitting)"]
        CHK --> EMB["Embedding Node\n(NVIDIA nv-embedqa-e5-v5)"]
        EMB --> VDB[("ChromaDB\nVector Store")]
    end
```

### Node Responsibility Matrix

| Node | Purpose | Input | Output | Error Fallback |
|------|---------|-------|--------|----------------|
| **Query Decomposer** | Splits raw query into semantic, statutory, procedural focuses | `query` | `decomposed_query` | Falls back to raw query as semantic focus |
| **Qualifier** | Classifies intent, domain, scenario type | `query` | `law_domain`, `is_scenario`, `requires_case_law`, `is_general_chat` | Defaults to `General` domain, no case law |
| **Retriever** | Hybrid dense+sparse search with RRF fusion | `query`, `decomposed_query`, `iteration_count` | `documents` (top 20 chunks) | Tries fallback embedding model, then returns empty |
| **Re-Ranker** | LLM filters irrelevant chunks | `query`, `documents` | `documents` (filtered) | Returns all documents unfiltered |
| **Grader** | Evaluates document sufficiency | `query`, `documents`, `requires_case_law`, `iteration_count` | `search_required`, `retry_retrieval` | Routes to generator with whatever context exists |
| **Rewriter** | Optimizes query for better retrieval | `query`, `iteration_count` | Rewritten `query`, incremented `iteration_count` | Keeps original query, still increments count |
| **Web Search** | DuckDuckGo search for case laws | `query`, `requires_case_law` | `case_laws` | Returns empty list |
| **Generator** | Synthesizes final legal response | `query`, `documents`, `case_laws`, `memory_summary`, `is_scenario` | `generation` | Returns error message |

---

##  Getting Started (Docker Deployment)

The easiest way to run the entire Ma'at stack (Frontend + FastAPI + AI Agents) is via the unified single-container Docker build. 

### Prerequisites
* Docker installed and running
* An `.env` file in the root directory containing your API keys (e.g., `NVIDIA_API_KEY`, etc.)

### 1. Build the Application
Ensure you include the period `.` at the end of the command!
```bash
sudo docker build -t maat-ai .
```

### 2. Run the Application
We mount the `data` and `vector_store` directories as live volumes so that your AI's brain and chat history are safely persisted on your local machine, not trapped inside the container!
```bash
sudo docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/vector_store:/app/vector_store \
  --env-file .env \
  maat-ai
```

### 3. Open the App
Navigate to **`http://localhost:8000`** in your browser. The single container serves the beautiful React UI and handles the backend API routing seamlessly!

---

##  Roadmap & Future Improvements

We are constantly iterating to make Ma'at more powerful and robust. Current efforts include:

* ✅ ~~**Hybrid Search (BM25 + Dense + RRF):**~~ *Shipped.* Retrieval now fuses keyword and semantic results via Reciprocal Rank Fusion.
* ✅ ~~**Self-Corrective RAG Loop:**~~ *Shipped.* Grader → Rewriter → Retriever retry loop with max-iteration circuit breaker.
* 🔄 **Hallucination Guard Node:** Post-generation verification that cited sections actually appear in the provided context.
* 🔄 **Tool Calling Agent Middleware:** Decoupling tool execution logic from primary agents to allow for real-time external API requests (e.g., live legal database scraping).
* 🔄 **Automated Legal Form Generation:** Integrating the LLM outputs dynamically into hardcoded HTML/Markdown templates to spit out production-ready legal forms.
* 🔄 **Expanded Case Law Analysis:** Deepening the Case Law Agent's capability to cross-reference conflicting judicial precedents.

---

##  How to Contribute

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. **Fork the Project**
2. **Create your Feature Branch:** `git checkout -b feature/AmazingFeature`
3. **Commit your Changes:** `git commit -m 'Add some AmazingFeature'` (Make sure you pass all PEP 8 Linting checks!)
4. **Push to the Branch:** `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Code Standards
* **Python Code:** Must strictly adhere to **PEP 8** style guidelines and aim for a 10/10 Pylint score.
* **No Hallucinations:** RAG outputs must be configured with strictly typed JSON schemas (Pydantic). 
* **Type Hinting:** Mandatory for all backend Python functions.

<div align="center">
  <p>Built with ❤️ for the Legal Tech Community.</p>
</div>
