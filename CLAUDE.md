# Project: Legal RAG Bot & Form Generator 

## 1. THE "WHY" ENCODING (INTENT & VALUES)

Core Value: Excellence and Evolution. Having established and validated our MVP foundation, we now focus on enhancing capabilities, expanding features, and improving performance while maintaining our commitment to stability and legal accuracy.

Architecture Intent: Stability with Enhancement. We maintain our strong Template-Based RAG foundation while thoughtfully extending capabilities through additional model options and refined agent behaviors.

Domain Constraint: Absolute strict adherence to provided legal context. Zero legal hallucination is permitted.

## 2. HARD RULES VS. PREFERENCES

$$BINARY BOUNDARIES: YOU MUST DO THIS$$

API Contract: Frontend (React) and Backend (FastAPI) MUST communicate exclusively via REST endpoints.

RAG Outputs: LLM extractions for form generation MUST use response_format={ "type": "json_object" } or LangChain structured output.

Context Isolation: If a legal query cannot be answered by the vector DB context, you MUST output: "I do not have enough information to answer this based on the provided documents."

Strict Typing: Python code MUST use explicit Type Hints (def process(query: str) -> dict:). FastAPI MUST use Pydantic models for request/response validation.

Python Coding Standards (PEP 8 & Pylint):

PEP 8 Compliance: All backend Python code MUST strictly adhere to the PEP 8 style guide.

Pylint Enforcement: Write code that yields a 10/10 Pylint score.

Naming Conventions: You MUST use snake_case for variables/functions, and PascalCase for classes.

Error Handling: You MUST use try-except blocks around all OpenAI API calls and ChromaDB queries. Return proper HTTP 500 status codes from FastAPI if the LLM fails.

$$ GUIDEPOSTS: YOU SHOULD PREFER THIS$$

UI Simplicity: Prefer standard React useState and basic CSS over complex state managers (no Redux) or heavy component libraries.

Speed: Prefer simple, readable loops and standard library functions over clever, heavily abstracted class hierarchies.

Error Handling: Prefer returning clean HTTP 500/400 errors from FastAPI with human-readable messages so the frontend can degrade gracefully.

## 3. ANTI-PATTERNS (THE "NEVER" LIST)

NEVER hallucinate Python libraries or Node packages. If it is not in requirements.txt or package.json, ask before using pip install or npm install.

NEVER write legal contracts from scratch via LLM text generation. (This breaks formatting and introduces risk).

NEVER rewrite working legacy code or refactor whole files unless explicitly instructed to do so. ONLY modify the lines necessary for the requested feature.

NEVER hardcode API keys (OPENAI_API_KEY). Always use os.environ.get() or a .env file.

NEVER leave raw // TODO: or # TODO: comments in generated code. Implement the fix or ask the user for clarification.

NEVER break the API schema. If you change a Pydantic model in FastAPI, you MUST update the corresponding TypeScript interface in React.

NEVER bypass Pylint rules. Avoid writing code that requires disabling Pylint warnings via # pylint: disable= unless absolutely unavoidable (e.g., framework-specific quirks).

## 4. PROGRESSIVE DISCLOSURE & POINTERS

This file contains top-level architectural rules. For specific subsystem contexts, check the following locations before modifying code:

Frontend specific rules: Check app/README.md (if it exists).

State Schema: Always read `server/agent/state.py` to understand the `AgentState` TypedDict before adding new fields to graph state.

## 5. INTENT-BASED VERIFICATION ROUTINE

Before presenting final code or confirming a task is complete, you MUST execute this self-correction loop:

Linting Check: Does the Python code pass PEP 8? (Mentally verify against Pylint standards or run pylint target_file.py). Ensure 10/10 score.

Type Check: Do the FastAPI endpoint Pydantic schemas perfectly match the React frontend fetch payloads?

CORS Check: If adding a new API endpoint, is it covered by the CORS middleware in main.py?

State Check: If modifying React state, verify it does not cause an infinite render loop in a useEffect.

## 6. TECH STACK & COMMANDS

Frontend: React, TypeScript, HTML/CSS (Vite) -> cd app && npm run dev

Backend: FastAPI (Python 3.11+) -> cd server && uvicorn api.main:app --reload

AI/DB Stack: Nvidia Nim API (OpenAI-compatible), LangChain, LangGraph, Pinecone

Linting: Pylint -> cd server && pylint api/ agent/

## 7. DIRECTORY STRUCTURE
/
├── app/                        # React Frontend (Vite)
│   ├── src/
│   │   ├── api.ts              # Fetch wrappers for FastAPI endpoints
│   │   ├── components/         # ChatArea, ChatInput, Sidebar, SplashScreen
│   │   └── App.tsx             # Main Layout
├── server/                     # Backend Services
│   ├── api/                    # FastAPI Application
│   │   ├── main.py             # FastAPI application & REST endpoints
│   │   ├── models.py           # Pydantic models for request/validation
│   │   ├── routes.py           # API route definitions
│   │   └── security.py         # Authentication and security utilities
│   └── agent/                  # AI Agent Logic (LangGraph)
│       ├── chat_graph.py       # LangGraph orchestration (graph compilation)
│       ├── model.py            # Embedding & Chat Model initialization (NVIDIA NIM)
│       ├── state.py            # LangGraph TypedDict definitions (State schema)
│       ├── node/               # Individual LangGraph Nodes
│       │   ├── chunking.py     # Markdown document chunking
│       │   ├── cleaning.py     # Markdown text cleaning
│       │   ├── embedding.py    # Document embedding and vector storage
│       │   ├── generator.py    # Final legal synthesis and response generation
│       │   ├── grader.py       # Document relevance grading
│       │   ├── ingestion.py    # PDF document ingestion and processing
│       │   ├── qualifier.py    # Query intent and scenario classification
│       │   ├── query_decomposer.py # Query decomposition for hybrid retrieval
│       │   ├── reranker.py     # Document re-ranking for relevance filtering
│       │   ├── retriever.py    # Vector database retrieval
│       │   ├── rewriter.py     # Query rewriting for better retrieval
│       │   └── web_search.py   # External web search for case law
│       ├── prompt/             # LangChain Prompt Templates
│       │   ├── generator_prompt.py
│       │   ├── grader_prompt.py
│       │   ├── qualifier_prompt.py
│       │   ├── query_decomposer_prompt.py
│       │   ├── reranker_prompt.py
│       │   ├── rewriter_prompt.py
│       │   ├── search_query_prompt.py
│       │   └── __init__.py
│       └── utils/              # Utility Functions
│           ├── logger.py       # Centralized logging configuration
│           ├── embedding_utils.py # Vector database helper functions
│           ├── chunking_utils.py # Markdown text chunking utilities
│           ├── cleaning_utils.py # Text cleaning and normalization
│           └── ingestion_utils.py # Document processing helpers
├── data/                       # Raw PDF source documents
└── vector_store/              # Persistent Pinecone vector database (cloud) + BM25 cache
├── requirements.txt           # Python backend dependencies
├── package.json               # Frontend Node.js dependencies
├── Dockerfile                 # Containerization configuration
└── .env                       # Environment variables (API keys, configuration)

## 8. CURRENT ARCHITECTURE (LangGraph)

The chat pipeline is defined by `server/agent/chat_graph.py` (compile with `build_chat_graph()`). The state schema lives in `server/agent/state.py`. Graph nodes live in `server/agent/node/` and include: `query_decomposer`, `qualifier`, `retriever`, `reranker`, `grader`, `rewriter`, `web_search`, `generator`.

For the canonical pipeline diagram and node-by-node responsibility matrix, see `README.md §System Architecture`.

### Component Notes
1. **State Definition (`server/agent/state.py`)** — `AgentState` TypedDict with fields including `session_id`, `chat_history`, `memory_summary`, `query`, `is_scenario`, `requires_case_law`, `documents`, `case_laws`, `generation`, `iteration_count`, plus `ingest_*` fields for the ingestion graph.
2. **Self-Corrective RAG (`server/agent/node/`)** — `grader` evaluates retrieved docs; if `retry_retrieval` is set, request loops through `rewriter` → `retriever`. `generator` enforces strict context adherence.
3. **Hybrid Retrieval** — Dense (NVIDIA Nemotron embeddings via Pinecone) + Sparse (BM25 in-memory, rebuilt from Pinecone, cached to disk). Fused via RRF (`server/agent/node/retriever.py`).
4. **Chat Session Persistence** — Sessions are persisted as JSON files at `data/chats/{session_id}.json` by `server/api/routes.py`. No separate `history.py` module exists.
5. **Model Configuration** — Chat and embedding models configurable via environment variables (see `.env.example`).
