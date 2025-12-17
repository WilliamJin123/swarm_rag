# SwarmRAG Research Project Specifications

## Datasets

### Primary
- **STARK** *(currently pursuing)* - Multi-hop reasoning benchmark

### Secondary Candidates
- **HotpotQA** - Multi-hop QA with Wikipedia articles
- **2WikiMultiHopQA** - Structured comparison questions
- **MuSiQue** - Compositional reasoning, no shortcuts
- **Natural Questions (NQ)** - Real Google user queries
- **MS MARCO** - Large-scale IR with Bing queries
- **BEIR Benchmark** - Diverse out-of-domain evaluation

### Domain-Specific (Optional)
- **PubMedQA** - Biomedical
- **FinQA** - Financial reasoning
- **MultiRC** - Multi-sentence reading comprehension

## Persistence Strategy

### Recommended: Local-First
- **FAISS** - Vector similarity search
- **NetworkX** - Graph relationships
- **NumPy** - Raw embedding storage
- **DiskANN** - Alternative for very large scale

<!-- **Advantages:** Full control, no API costs, reproducibility, fast iteration

### Cloud Options (Post-Research)
- **Pinecone** - Managed vector DB
- **Qdrant** - Open-source vector search
- **Weaviate** - Vector DB with graph capabilities

**Use when:** Production deployment, remote collaboration, >100GB embeddings -->

## Embedding Configuration

### Model: Gemini Embeddings 001

<!-- **Recommended Dimension: 768**
- Optimal balance of expressiveness vs. speed
- Sufficient for most retrieval tasks
- Efficient for multiple similarity computations -->

### Ablation Study Plan
1. Baseline: **768**
2. Compare: **384** (faster) vs **1536** (more expressive)
3. Measure: Retrieval quality / computational cost

<!-- ### Dimension Guidelines
- **384** - If speed is critical
- **768** - Default recommendation
- **1536** - Highly specialized domains, long documents
- **3072** - Avoid unless small dataset (<10K docs) + GPU resources -->