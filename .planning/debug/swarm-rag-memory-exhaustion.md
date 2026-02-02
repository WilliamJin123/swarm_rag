---
status: diagnosed
trigger: "Swarm RAG system exhausts system memory, causing crashes. Memory should be a few GBs max with lazy-loaded queries."
created: 2026-01-31T18:30:00Z
updated: 2026-01-31T19:20:00Z
---

## Current Focus

hypothesis: CONFIRMED - Multiple copies of embedding data loaded into memory simultaneously
test: Static code analysis complete
expecting: Memory exhaustion from: (1) doc_embs dict retained, (2) TorchVectorStore.from_dict() creates full copies, (3) StarkPreComputedEmbeddingHandler copies all query_embs to GPU
next_action: Document root cause for fix planning

## Symptoms

expected: Few GBs max total memory (VRAM + CPU), queries should lazy load on-demand
actual: System crashes/freezes due to memory exhaustion, Claude Code session crashes
errors: No explicit OOM errors - system runs out of allocatable memory
reproduction: Running swarm RAG scripts causes memory exhaustion
started: After recent GSD changes - was working before Phase 1 of project plan

## User Hypotheses

1. Memory leak - something allocating without releasing
2. Naive loading - loading huge things into memory at once
3. Queries not lazy loaded - all queries loaded at once instead of on-demand
4. Cache design issue - preallocating too much memory
5. Cache prefill issue - prefilling cache incorrectly

## Eliminated

- hypothesis: Cache prefill issue
  evidence: EmbeddingCache and FitnessCache have LRU eviction (maxsize parameter), fitness cache stores only floats
  timestamp: 2026-01-31T18:45:00Z

- hypothesis: Cache design issue (preallocation)
  evidence: Caches use dictionaries, not preallocated tensors. LRU eviction is implemented.
  timestamp: 2026-01-31T18:50:00Z

## Evidence

- timestamp: 2026-01-31T18:40:00Z
  checked: stark/load_stark.py - load_and_download_embeddings()
  found: Uses torch.load() with mmap=True, returns query_embs and doc_embs as dictionaries
  implication: mmap should reduce initial memory, but downstream code creates copies

- timestamp: 2026-01-31T18:45:00Z
  checked: swarm_rag_module/swarm_rag/integrations/torch_vector_store.py - from_dict()
  found: Lines 176-190 iterate through ALL embeddings, create emb_list, then torch.stack(). Creates full copy in CPU RAM before moving to device.
  implication: For prime (129K docs * 1536 dim * 4 bytes = 792MB), this creates ~792MB CPU copy, then another ~792MB GPU copy

- timestamp: 2026-01-31T18:50:00Z
  checked: swarm_rag_module/swarm_rag/integrations/stark.py - StarkPreComputedEmbeddingHandler.__init__
  found: Lines 268-273 iterate through ALL query embeddings and copy each to GPU device. No device param passed in evolve_stark.py line 134.
  implication: ALL query embeddings copied to GPU, original dict retained

- timestamp: 2026-01-31T18:55:00Z
  checked: stark/evolve_stark.py - run_evolution()
  found: Line 119 loads query_embs/doc_embs, line 127 creates StarkVectorStore (copies doc_embs), line 134 creates StarkPreComputedEmbeddingHandler (copies query_embs). Original dicts never explicitly deleted.
  implication: Memory contains: (1) original mmap'd dicts, (2) TorchVectorStore copy, (3) StarkPreComputedEmbeddingHandler copy

- timestamp: 2026-01-31T19:00:00Z
  checked: swarm_rag_module/swarm_rag/core/swarm_retriever.py - _init_multi_query_state()
  found: Lines 2849-2852 allocate query_pheromones tensor of shape (batch_size, n_nodes). With batch_size=100 and n_nodes=129K for prime, this is 100*129000*4=51.6MB per batch. For amazon (957K nodes): 382MB per batch.
  implication: Additional per-batch memory allocation, but smaller than embedding copies

- timestamp: 2026-01-31T19:05:00Z
  checked: Memory calculation for prime dataset
  found: doc_embs = 129K * 1536 * 4 bytes = 792MB. With dict overhead + multiple copies: easily 2-3GB CPU + 1GB+ GPU before any processing starts.
  implication: For amazon (957K docs): ~5.9GB for doc embeddings alone, multiplied by copies = 10-15GB+

- timestamp: 2026-01-31T19:10:00Z
  checked: TorchVectorStore dense mode
  found: Line 111-114: Dense mode allocates torch.zeros((max_id, dim)) which for prime = 129K*1536*4=792MB additional GPU memory
  implication: Dense mode doubles GPU memory usage

## Resolution

root_cause: MULTIPLE MEMORY COPIES AND EAGER LOADING

1. **Primary Issue - TorchVectorStore.from_dict() inefficient copy**:
   - File: `swarm_rag_module/swarm_rag/integrations/torch_vector_store.py`
   - Lines 176-190: Creates list of ALL embeddings, then torch.stack()
   - Creates ~792MB temporary CPU allocation for prime dataset

2. **Secondary Issue - Original dicts never released**:
   - File: `stark/evolve_stark.py`
   - Line 119: `query_embs, doc_embs = load_and_download_embeddings(args.dataset)`
   - These dicts remain in scope for entire run, even after being copied to stores

3. **Tertiary Issue - StarkPreComputedEmbeddingHandler copies all to GPU**:
   - File: `swarm_rag_module/swarm_rag/integrations/stark.py`
   - Lines 268-273: Eagerly copies ALL query embeddings to GPU at init

4. **Quaternary Issue - No streaming/lazy loading**:
   - All embeddings loaded at once, no chunked loading or memory-mapped tensor access

MEMORY FLOW (prime dataset, CUDA):
- Load: 792MB (mmap'd doc_embs) + smaller query_embs
- TorchVectorStore.from_dict(): +792MB CPU (stack), +792MB GPU
- StarkPreComputedEmbeddingHandler: +query_embs on GPU
- Dense mode (if enabled): +792MB GPU
- Total: 2-4GB+ before any queries run

For amazon (957K docs, ~6GB per copy): System easily exceeds available RAM/VRAM.

fix: (Not applied - find_root_cause_only mode)
verification:
files_changed: []
