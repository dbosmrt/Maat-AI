# Contributing to Ma'at

First off, thank you for considering contributing to Ma'at! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

Ma'at is a sophisticated legal advisory AI assistant designed to provide accurate, context-aware, and strictly factual legal guidance. Because we deal with legal intelligence, our core philosophy is **Stability over Features** and **Zero Hallucination Tolerance**. 

Please review this document to understand our strict coding standards and workflows before submitting a Pull Request.

## Core Philosophy

1. **Strict Context Adherence:** We use a template-based RAG approach. The AI extracts JSON; the backend injects it. Zero legal hallucination is permitted. If the context does not answer a query, the system MUST state it does not have enough information.
2. **Demo Survival & Stability:** A simple, working MVP is infinitely more valuable than a complex, broken app.
3. **No LLM Output Generation for Code/Contracts:** Never write legal contracts from scratch via LLM text generation. Always use strict JSON extraction and structured outputs.

## Development Setup

### 1. Local Python Setup
We use Python 3.11+. We recommend setting up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory based on `.env.example` (if present) or ensure you have your API keys set:
- `NVIDIA_NIM_KEY`: Required for embeddings and LLM generation.

### 3. Running Locally
- **Backend (FastAPI):** `cd server && uvicorn main:app --reload`
- **Frontend (Vite/React):** `cd app && npm install && npm run dev`

### 4. Docker (Recommended)
You can build and run the entire stack using Docker:
```bash
sudo docker build -t maat-ai .
sudo docker run -d -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/vector_store:/app/vector_store --env-file .env maat-ai
```

## Contribution Workflow

1. **Fork the Repository:** Create your own fork of the project.
2. **Create a Branch:** Use a descriptive branch name based on the type of work:
   - Features: `feature/AmazingFeature`
   - Bug Fixes: `bugfix/IssueName`
3. **Make your Changes:** Follow the strict coding standards outlined below.
4. **Commit your Changes:** Use clear, descriptive commit messages.
5. **Push to your Branch:** `git push origin feature/AmazingFeature`
6. **Open a Pull Request:** Describe the changes thoroughly. Ensure all CI checks (GitHub Actions) pass.

## Strict Coding Standards

We enforce strict rules for all contributions. **Pull requests failing these checks will not be merged.**

### Python & Backend Standards (FastAPI)
- **PEP 8 Compliance:** All backend Python code MUST strictly adhere to the PEP 8 style guide.
- **Pylint Score:** Code must yield a 10/10 Pylint score. Avoid disabling Pylint warnings (`# pylint: disable=`) unless absolutely necessary (e.g., framework quirks).
- **Type Hinting:** Explicit Type Hints are mandatory (e.g., `def process(query: str) -> dict:`).
- **Pydantic Validation:** FastAPI must use Pydantic models for request/response validation. If you change a backend Pydantic model, you MUST update the corresponding frontend TypeScript interface.
- **Error Handling:** Use `try-except` blocks around all OpenAI/Nvidia API calls and ChromaDB queries. Return proper HTTP 500/400 status codes with human-readable messages.

### Frontend Standards (React/Vite)
- **UI Simplicity:** Prefer standard React `useState` and basic CSS over complex state managers (like Redux) or heavy component libraries.
- **API Contract:** Frontend and Backend MUST communicate exclusively via REST endpoints.

### General Anti-Patterns (The "Never" List)
- **NEVER** hallucinate Python libraries or Node packages. Only use dependencies listed in `requirements.txt` or `package.json`.
- **NEVER** hardcode API keys. Always use `os.environ.get()` or a `.env` file.
- **NEVER** leave raw `// TODO:` or `# TODO:` comments in code. Implement the fix or ask for clarification in an issue.
- **NEVER** rewrite working legacy code or refactor whole files unless explicitly instructed to do so. Only modify what is necessary.

## Continuous Integration (GitHub Actions)

When you submit a PR, the following automated checks will run:
1. **Linter & Type Checker:** Validates PEP 8, Pylint (10/10), and MyPy typing.
2. **Unit Tests:** Runs the `pytest` test suite located in the `test/` directory.
3. **Docker Build:** Verifies that the container builds successfully without errors.

Thank you for contributing to making Ma'at a reliable, factual, and robust Legal RAG AI!
