# Codebase Concerns

**Analysis Date:** 2026-01-29

## Tech Debt

**Hardcoded Weight Constants in Swarm Movement:**
- Issue: Movement strategy weights are hardcoded to 0.3 (semantic), 0.4 (centrality), 0.3 (repulsion) in retrieval logic with TODO comment
- Files: `swarm_rag_module/swarm_rag/core/swarm_retriever.py` (line 2114)
- Impact: Cannot adjust movement strategy balance without code changes; reduces flexibility for domain-specific tuning
- Fix approach: Extract weights to `SwarmRetriever` configuration parameters, make them genome-evolvable in strategy definitions

**Manual Strategy Compilation Pipeline:**
- Issue: Genomes must be compiled to kwargs before passing to `SwarmRetriever.retrieve_batch()`. Multiple compilation steps with implicit state passing
- Files: `swarm_rag_module/swarm_rag/evolution/types/genome.py`, `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py`
- Impact: Error-prone integration; easy to miscompile strategies or forget compilation step
- Fix approach: Add builder pattern API (partially addressed in recent commits) to encapsulate compilation logic

**Broad Exception Handling in Critical Paths:**
- Issue: Many `except Exception` blocks silently catch and log errors without specific error types (lines 1278-1287 in swarm_retriever.py, 514-516)
- Files: `swarm_rag_module/swarm_rag/core/swarm_retriever.py` (heuristic evaluation, query failures), `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py`
- Impact: Hides real errors (network failures, model crashes) behind generic logging; makes debugging difficult; masks data quality issues
- Fix approach: Catch specific exceptions (ValueError, ConnectionError, RuntimeError) and implement per-error handling strategies

**Dual-Mode Genome Architecture Complexity:**
- Issue: Codebase supports two genome modes: `expression_tree` (symbolic strategies) and `weighted_sum` (linear combinations), with separate mutation/crossover paths
- Files: `swarm_rag_module/swarm_rag/evolution/execution/strategies.py`, `swarm_rag_module/swarm_rag/evolution/execution/weighted_sum.py`
- Impact: Increased maintenance burden; some operators only work with one mode; difficult to add new operators
- Fix approach: Consider deprecating one mode or creating unified operator interface that works transparently with both

**Memory Unbounded in Interactive Sessions:**
- Issue: `StepProfiler` uses rolling window but no cap on history per section in degenerate cases; pheromone hashtables can grow without bounds
- Files: `swarm_rag_module/swarm_rag/core/swarm_retriever.py` (lines 41-43), pheromone storage
- Impact: Long-running evaluations may accumulate unnecessary memory; potential memory leak in production
- Fix approach: Implement strict size caps for pheromone hashtables and add memory monitoring to profiler

## Known Bugs

**Group Ratio Mutation Not Applied in Aggressive Strategy:**
- Symptoms: Test `test_aggressive_mutation_ignores_ratios` expects ratios to change during aggressive mutation but they may not
- Files: `swarm_rag_module/swarm_rag/evolution/execution/strategies.py` (line 905-909), test file `swarm_rag_module/tests/bugs/reproduce_ratio_mutation_bug.py`
- Trigger: Call `GeneticStrategies.aggressive_mutation()` and inspect `group_ratios` dict
- Current status: Code appears to have the ratio mutation loop (lines 906-909) but test comments suggest historical bug
- Fix approach: Verify ratio mutation works in both `aggressive_mutation` and `guided_mutation`; add regression test with assertion on changed ratios

**Early Exit Threshold Edge Cases:**
- Symptoms: Quarter checkpoint (25% of queries) may not filter genomes effectively if quality threshold is too high/low
- Files: `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py` (line 38: DEFAULT_EARLY_EXIT_THRESHOLD = 0.30)
- Trigger: Run evolution with extremely small or large query sets where quarter point is very small
- Workaround: Manually adjust `early_exit_threshold` in config
- Current mitigation: Hardcoded to 0.30 quality threshold; only tested with typical dataset sizes

## Security Considerations

**Unvalidated Genome Parameter Ranges:**
- Risk: Genome parameters accept arbitrary values during crossover/mutation without range validation; potential for invalid configurations
- Files: `swarm_rag_module/swarm_rag/evolution/types/genome.py`, `swarm_rag_module/swarm_rag/evolution/execution/strategies.py` (mutation functions)
- Current mitigation: Some parameters clamped during mutation (`max(min_v, min(max_v, value))`), but not all
- Recommendations: Validate all genome parameters in `Genome.__post_init__` and after every mutation; add type annotations for valid ranges

**Unencrypted Checkpoint Serialization:**
- Risk: Evolution checkpoints saved to disk may contain sensitive information from training data (query embeddings, ground truth labels)
- Files: `swarm_rag_module/swarm_rag/evolution/storage/run_manager.py` (line 472)
- Current mitigation: None detected; assumes secure filesystem
- Recommendations: Add optional checkpoint encryption, document sensitivity of checkpoint files, restrict file permissions

**External API Key Exposure in Logs:**
- Risk: LLM client calls to external models (Gemini, Cohere) may log API keys in debug mode
- Files: `swarm_rag_module/swarm_rag/evolution/llm/client.py`, `swarm_rag_module/swarm_rag/integrations/gemini_embed.py`, `swarm_rag_module/swarm_rag/integrations/cohere_embed.py`
- Current mitigation: Reliance on Python logging config not including auth headers
- Recommendations: Explicitly mask credentials in HTTP client logs; validate env vars at startup, not at call time

## Performance Bottlenecks

**Large Swarm Retriever File (3097 lines):**
- Problem: Single file implements state management, movement logic, pheromone operations, heuristic evaluation, and batching
- Files: `swarm_rag_module/swarm_rag/core/swarm_retriever.py`
- Cause: Feature accumulation without refactoring; high complexity in single module makes local reasoning difficult
- Improvement path: Split into `state_manager.py`, `movement_logic.py`, `pheromone_ops.py`, `heuristics_wrapper.py` to improve testability and maintenance

**Inefficient Expression Tree Evaluation:**
- Problem: Expression trees evaluated interpretively for each genome; no caching of subtree results across similar genomes
- Files: `swarm_rag_module/swarm_rag/evolution/types/expressions.py` (tree evaluation), used in evaluator
- Cause: Early-stage optimization; interpreter overhead compounds with population size
- Improvement path: Implement subtree memoization or compile trees to PyTorch operations for batch evaluation

**Sequential Query Evaluation in Population Evaluator:**
- Problem: Even with shared precompute, genomes evaluated sequentially within thread pool; could batch across genomes for multi-query GPU
- Files: `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py` (lines 224-232)
- Cause: Genome-level parallelism chosen over query-level; shared context not used for cross-genome batching
- Improvement path: Implement cross-genome batching where multiple genomes process same query batch concurrently

## Fragile Areas

**Heuristic Registry Dynamic Registration:**
- Files: `swarm_rag_module/swarm_rag/core/heuristics.py`, `swarm_rag_module/swarm_rag/interfaces/registry.py`
- Why fragile: Custom heuristics added via decorator pattern; if registry not initialized before genome evaluation, missing functions cause silent failures
- Safe modification: Always call `HeuristicRegistry.register_*()` in module `__init__`, test registry state before evaluation
- Test coverage: No dedicated test for registry initialization order; integration tests may miss missing heuristics

**Genome Compilation Cache:**
- Files: `swarm_rag_module/swarm_rag/evolution/types/genome.py` (line 502: `clear_cache()`)
- Why fragile: Cache invalidation on mutation may miss edge cases; if `clear_cache()` not called after direct strategy modification, compiled version stale
- Safe modification: Never directly modify `genome.strategies`; always use mutation operators that call `clear_cache()`
- Test coverage: No test validating cache correctness after strategy mutation

**Pheromone Hashtable Collisions:**
- Files: `swarm_rag_module/swarm_rag/core/swarm_retriever.py` (lines 2137-2138: pheromone deposit via hash)
- Why fragile: Hash-table implementation of pheromone storage; collision handling not documented, may degrade with large graph sizes
- Safe modification: Keep node ID ranges small, monitor hash collision rates in profiler
- Test coverage: No test for pheromone correctness under high node count; may not work correctly with graphs >100k nodes

**Early Exit Checkpoint Threshold Sensitivity:**
- Files: `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py` (lines 37-38, 205-207)
- Why fragile: Single hardcoded threshold (0.30) applied to all quality scores; no adaptation to dataset characteristics
- Safe modification: Add config option for threshold; validate threshold during initialization
- Test coverage: Early exit only tested with mid-size datasets; edge cases with <10 queries or >10k queries untested

**LLM-Guided Mutation Decision Caching:**
- Files: `swarm_rag_module/swarm_rag/evolution/llm/decision_tracker.py`
- Why fragile: Decisions cached by (generation, genome_id) tuple; if genome ID changes or duplication happens, cache may be skipped
- Safe modification: Validate cache key uniqueness; add hash of genome state to key
- Test coverage: No test for cache correctness under genome cloning or duplication

## Scaling Limits

**Query Embedding Batch Size Fixed:**
- Current capacity: Default batch size of 32 for embedding
- Limit: GPU memory; 32 queries * embedding_dim (typically 1536) = ~200MB per batch
- Scaling path: Make batch size adaptive based on GPU memory availability; implement memory-aware batching

**Population Archive Size Unbounded:**
- Current capacity: MAP-Elites archive grows with generations; no pruning of old cells
- Limit: Memory for storing all elite genomes; with 1000x1000 grid and 100 generations, could have millions of stored genomes
- Scaling path: Implement archive compression, periodic elitism-based pruning, or secondary storage with lazy loading

**Pheromone Storage Linear in Graph Size:**
- Current capacity: Pheromone values stored per node; hashtable lookup O(1) but storage O(nodes)
- Limit: Large graphs (>1M nodes) will consume multiple GB for pheromone data
- Scaling path: Implement approximate pheromone (e.g., sketches or lossy compression), hierarchical storage

**Number of Heuristic Features Not Scalable:**
- Current capacity: Expression trees built from features; depth limit is 5, max features typically ~10
- Limit: Exponential tree space; adding 5+ new features makes tree search intractable
- Scaling path: Hierarchical feature organization, feature selection before evolution

## Dependencies at Risk

**torch.compile() Availability:**
- Risk: Code attempts to use `torch.compile()` for speed but silently falls back on failure
- Files: `swarm_rag_module/swarm_rag/core/swarm_retriever.py` (lines 216-223)
- Impact: If compilation fails in production, falls back to slow interpreted code without alerting
- Migration plan: Move torch.compile() to optional optimization layer; provide explicit control via config flag

**External Embedding Models (Gemini, Cohere):**
- Risk: Direct dependency on third-party embedding APIs; API changes or deprecations break code
- Files: `swarm_rag_module/swarm_rag/integrations/gemini_embed.py`, `swarm_rag_module/swarm_rag/integrations/cohere_embed.py`
- Impact: Service outages cause complete failure; no fallback mechanism
- Migration plan: Implement adapter pattern to support multiple embedding backends; add local/cached fallback

**LLM-Guided Evolution Optional Dependency:**
- Risk: LLM client initialization required even if LLM mutations disabled; API failures affect entire evolution
- Files: `swarm_rag_module/swarm_rag/evolution/llm/client.py`, `swarm_rag_module/swarm_rag/evolution/llm/constrained_executor.py`
- Impact: If LLM service unavailable, cannot run evolution at all (config.llm.enabled=False doesn't prevent initialization)
- Migration plan: Lazy-load LLM client only when first LLM mutation strategy used

## Missing Critical Features

**No Online Adaptation:**
- Problem: Genome parameters fixed after initialization; cannot adapt to changing data or performance feedback during retrieval
- Blocks: Real-time personalization, active learning scenarios, online reinforcement learning integration

**No Multi-Objective Explanation:**
- Problem: Pareto solutions selected but no explanation of trade-offs or hypervolume contribution
- Blocks: User-facing model selection, understanding quality-latency trade-offs

**No Fault Tolerance in Long Runs:**
- Problem: Multi-GPU evolution has no checkpoint-resume capability; days-long runs cannot be resumed after failure
- Blocks: Production deployment, large-scale evolutionary studies

**No Progressive Visualization:**
- Problem: Evolution results only observable at end; no streaming logs of archive evolution, fitness progression
- Blocks: Interactive tuning, real-time debugging, online monitoring

## Test Coverage Gaps

**Mutation Operator Coverage:**
- What's not tested: Interaction effects between mutation operators; ratio mutation under extreme values (0.01, 0.99); mutation under 100% mutation rate
- Files: `swarm_rag_module/tests/evolution/test_focused_mutation.py`, `swarm_rag_module/tests/evolution/test_parallel_mutation.py`
- Risk: Mutations may produce invalid genomes silently; genetic diversity loss undetected
- Priority: High - mutation is core evolutionary mechanism

**Early Exit Threshold Sensitivity:**
- What's not tested: Threshold behavior with <50 total queries; threshold behavior with >10k queries; interaction with multiple early exit checkpoints
- Files: `swarm_rag_module/tests/integration/test_full_evolution.py` doesn't include early exit variations
- Risk: Early exit may be completely ineffective or oversensitive in edge cases
- Priority: Medium - only affects large-scale deployments

**Error Recovery in Batch Evaluation:**
- What's not tested: Single query failure in batch; multiple concurrent failures; recovery behavior
- Files: `swarm_rag_module/tests/integration/` - no test for exception handling
- Risk: Partial failures silently return empty results; evaluation metrics become unreliable
- Priority: High - affects reliability

**Pheromone Correctness Under Concurrency:**
- What's not tested: Concurrent pheromone updates from multiple agents; pheromone values in concurrent retrieval batches
- Files: No concurrent pheromone test found
- Risk: Race conditions in multi-threaded retrieval may corrupt pheromone state
- Priority: Medium - only affects multi-query batching

**Cross-Genome Strategy Interference:**
- What's not tested: Does evaluating genome A affect fitness of genome B via shared state?; pheromone state leakage between genomes
- Files: No isolation test found
- Risk: Genome fitness scores depend on evaluation order; archive selection incorrect
- Priority: High - correctness issue

**Expression Tree Evolution Consistency:**
- What's not tested: Tree mutations produce valid Python expressions; tree compilation handles all node types; recursive strategy definitions
- Files: `swarm_rag_module/tests/evolution/test_genome.py` - basic coverage only
- Risk: Silent expression evaluation failures; strategy trees reduce to no-ops
- Priority: High - affects strategy correctness

---

*Concerns audit: 2026-01-29*
