# Technology Stack

**Analysis Date:** 2026-01-29

## Languages

**Primary:**
- Python 3.10 - All development, core algorithms, and scripts
- PyTorch/CUDA C++ - GPU acceleration for tensor operations and vector stores

**Secondary:**
- Bash/Shell - Utility scripts and environment setup

## Runtime

**Environment:**
- Python 3.10.19

**Package Manager:**
- pip
- Lockfile: `requirements.txt` (172 lines)

## Frameworks

**Core ML/AI:**
- PyTorch 2.9.1 - Tensor computation, GPU support, neural operations
- Transformers 4.57.3 - Transformer models and embeddings
- LangChain 1.2.0 - LLM orchestration and chains
- LangChain-Core 1.2.5 - Foundation layer for LangChain

**Vector & Graph Search:**
- FAISS-CPU 1.13.2 - Vector similarity search and indexing
- NetworkX 3.4.2 - Graph algorithms and analysis
- Torch-Geometric 2.7.0 - Graph neural network support

**Data Processing:**
- Pandas 2.3.3 - DataFrame operations
- NumPy 1.26.4 - Array operations
- SciPy 1.15.3 - Scientific computing
- Polars 1.36.1 - High-performance DataFrame library

**Dataset & ML Tools:**
- Datasets 4.4.2 - HuggingFace datasets loading
- Scikit-Learn 1.7.2 - Machine learning utilities
- Sentence-Transformers 5.2.0 - Embedding models

**Domain-Specific:**
- STaRK-QA 1.1.0 - Question answering dataset/benchmark
- ColBERT-AI 0.2.22 - Dense retrieval system
- NLTK 3.9.2 - Natural language processing
- Beautiful Soup 4 4.14.3 - HTML/XML parsing
- RDKit-PyPi 2022.9.5 - Chemistry toolkit (molecular data)

**Monitoring & Observability:**
- Sentry-SDK 2.48.0 - Error tracking and crash reporting
- Wandb 0.23.1 - Weights & Biases for experiment tracking
- Tenacity 9.1.2 - Retry logic and failure handling

**Web & Server:**
- Flask 3.1.2 - Web framework (if used)
- Tornado 6.5.4 - Async web server
- HTTPx 0.28.1 - Async HTTP client
- HTTPCORE 1.0.9 - HTTP primitives
- Werkzeug 3.1.4 - WSGI utilities

**Development & Testing:**
- Pytest 9.0.2 - Testing framework
- IPython 8.37.0 - Interactive shell
- Jupyter-Core 5.9.1 - Notebook runtime
- IPyKernel 7.1.0 - Jupyter kernel

**Async & Concurrency:**
- AIOHttp 3.13.2 - Async HTTP client/server
- AIOSignal 1.4.0 - Signal system for asyncio
- Async-Timeout 5.0.1 - Timeout handling for async
- Nest-Asyncio 1.6.0 - Nested event loop support
- AIOLIMITER 1.2.1 - Rate limiting for async operations

**Utilities & Infrastructure:**
- Python-Dotenv 1.2.1 - Environment variable loading
- Pydantic 2.12.5 - Data validation and settings
- Click 8.3.1 - CLI framework
- Rich 14.2.0 - Terminal formatting and tables
- TQDM 4.67.1 - Progress bars
- JobLib 1.5.3 - Caching and multiprocessing

**API Client Integration:**
- Anthropic 0.75.0 - Anthropic API client
- OpenAI 2.14.0 - OpenAI API client
- Cohere 5.20.1 - Cohere API client
- Google-GenerativeAI (genai) - Gemini API
- VoyageAI 0.3.7 - Voyage embedding API

## Key Dependencies

**Critical Infrastructure:**
- Anthropic 0.75.0 - LLM API for Claude models
- OpenAI 2.14.0 - LLM API for GPT models
- PyTorch 2.9.1 - Deep learning, GPU acceleration
- FAISS-CPU 1.13.2 - Vector search engine
- Transformers 4.57.3 - Pre-trained models and tokenizers

**Retrieval & RAG:**
- LangChain 1.2.0 - RAG framework and chains
- Sentence-Transformers 5.2.0 - Dense embeddings
- STaRK-QA 1.1.0 - Benchmark dataset
- ColBERT-AI 0.2.22 - Dense retrieval baseline

**Multi-LLM Provider Support (via keycycle):**
- Cerebras API (51 keys configured)
- OpenRouter (31 keys)
- Groq (16 keys)
- Gemini/Google (44 keys)
- Cohere (1 key)
- Mistral (1 key)
- CodeStral (1 key)

**Embedding Providers:**
- Cohere Embeddings (via `cohere` package)
- Gemini Embeddings (via `google-generativeai`)
- VoyageAI (via `voyageai` package)

**Data & Science:**
- Pandas 2.3.3 - Data manipulation
- NumPy 1.26.4 - Numerical computing
- SciPy 1.15.3 - Scientific algorithms
- Matplotlib 3.10.8 - Visualization

## Configuration

**Environment:**
- Loads from `.env` file via `python-dotenv`
- Device selection: `SWARM_RAG_DEVICE` (cuda or cpu)
- Profiling: `EVOLUTION_PROFILE`, `SWARM_PROFILE` flags (1 to enable)

**Key Configs Required:**
- Multiple API keys for LLM providers (Cerebras, Groq, OpenRouter, Gemini, Cohere)
- TiDB connection string: `TIDB_DB_URL` (MySQL-compatible)
- Provider selection for evolution system: `provider` and `model` parameters

**Build:**
- Setuptools (setuptools >= 61.0)
- Build backend: setuptools.build_meta
- Package installation: `pip install -e swarm_rag_module` (from `swarm_rag_module/pyproject.toml`)

## Platform Requirements

**Development:**
- Python 3.9+ (project requires 3.9, running 3.10)
- GPU support: CUDA-capable NVIDIA GPU (optional, falls back to CPU)
- 8GB+ RAM minimum
- 50GB+ disk space (for datasets, embeddings, evolution artifacts)

**Production:**
- Python 3.10 (tested version)
- CUDA Toolkit (for GPU support)
- Linux or Windows with CUDA support
- Containerization: Docker support implied by multi-stage requirements

## External Dependencies Not in requirements.txt

**keycycle Package** - Custom unified LLM client wrapper:
- Manages multiple LLM provider APIs (cerebras, openai, groq, etc.)
- Implements rotating API key management
- Handles rate limiting and provider-specific adapters
- Used by: `swarm_rag.evolution.llm.client.LLMClient`

**TiDB Cloud** - Vector database:
- MySQL-compatible connection at `TIDB_DB_URL`
- Used for storing/retrieving embeddings and metadata
- Protocol: `mysql+pymysql://`

---

*Stack analysis: 2026-01-29*
