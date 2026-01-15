# MAP-Elites Category Extension - Implementation Plan

## Context for New Sessions

This document contains a complete implementation plan for extending MAP-Elites with question categories. A new Claude Code session can use this to implement without needing additional exploration.

---

## Problem Statement

The current MAP-Elites implementation uses behavioral descriptors (like `aggressiveness`, `complexity`) that can overlap with fitness metrics optimized by Boltzmann selection. This creates redundancy and collapses diversity.

**Solution:** Use **question categories** (single-hop, multi-hop, entity type, etc.) as MAP-Elites dimensions instead. This enables evolution of **specialized retrieval strategies per category**.

---

## Current Architecture Overview

### Key Files (Read These First)

| File | Purpose |
|------|---------|
| `stark/load_stark.py` | Data loading, pattern for `precompute_stark_adjacency` (lines 74-131) |
| `stark/evolve_stark.py` | Main evolution entry point |
| `swarm_rag_module/swarm_rag/evolution/map_elites/archive.py` | Current single-archive implementation |
| `swarm_rag_module/swarm_rag/evolution/map_elites/descriptors/registry.py` | Pattern for extensible registry system |
| `swarm_rag_module/swarm_rag/evolution/orchestrators/map_elites.py` | Current MAP-Elites orchestrator |
| `swarm_rag_module/swarm_rag/evolution/execution/evaluator.py` | Population evaluation |
| `swarm_rag_module/swarm_rag/evolution/types/config.py` | Configuration TypedDicts |

### Current MAP-Elites Flow

1. `MapElitesArchive` stores genomes in a grid indexed by behavioral descriptors
2. Each cell holds the best-fitness genome for that behavioral region
3. Parent selection is random from archive (not fitness-based)
4. Fitness (`quality_score`) is used only for within-cell comparison
5. Descriptors are calculated via `FlexibleDescriptorCalculator` using `DescriptorRegistry`

### STARK Datasets

Three datasets: `prime` (medical), `amazon` (e-commerce), `mag` (academic)
- No existing question categorization
- All questions treated uniformly during evolution
- `answer_ids` length varies (proxy for single/multi-hop)

---

## Design Decisions (User Confirmed)

1. **Category Definition:** Manual annotation via pre-computation script (like `precompute_stark_adjacency`)
2. **Grid Structure:** **Separate archives** per question category (specialist genomes)
3. **Built-in Categories:** Hop complexity, entity type, retrieval density, query complexity
4. **Extensibility:** Registry pattern for adding new category analyzers

---

## Implementation Plan

### Phase 1: Category Analyzer System

**Create `stark/categories/` directory structure:**

```
stark/categories/
  __init__.py
  types.py           # CategoryValue, QACategoryAssignment, CategoryCache
  analyzers/
    __init__.py
    base.py          # CategoryAnalyzer ABC
    registry.py      # CategoryAnalyzerRegistry
    builtin.py       # Built-in analyzers
```

**types.py - Core Types:**

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple

class CategoryType(Enum):
    DISCRETE = auto()    # e.g., single-hop, multi-hop
    CONTINUOUS = auto()  # e.g., difficulty score 0.0-1.0
    ORDINAL = auto()     # e.g., hop_count: 1, 2, 3, 4+

@dataclass
class CategoryValue:
    name: str
    category_type: CategoryType
    discrete_label: Optional[str] = None
    continuous_value: Optional[float] = None
    ordinal_value: Optional[int] = None
    confidence: float = 1.0

@dataclass
class QACategoryAssignment:
    qa_index: int
    query_id: Any
    categories: Dict[str, CategoryValue] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CategoryCache:
    dataset_name: str
    version: str = "1.0"
    n_samples: int = 0
    category_schema: Dict[str, Dict] = field(default_factory=dict)
    assignments: List[QACategoryAssignment] = field(default_factory=list)
    _by_category: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)

    def get_indices_by_category(self, category_name: str, value: str) -> List[int]:
        return self._by_category.get(category_name, {}).get(value, [])

    def build_indices(self):
        """Build reverse index after loading."""
        self._by_category = {}
        for assignment in self.assignments:
            for cat_name, cat_val in assignment.categories.items():
                if cat_name not in self._by_category:
                    self._by_category[cat_name] = {}
                key = cat_val.discrete_label or str(cat_val.ordinal_value) or "continuous"
                if key not in self._by_category[cat_name]:
                    self._by_category[cat_name][key] = []
                self._by_category[cat_name][key].append(assignment.qa_index)
```

**analyzers/base.py - ABC:**

```python
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from ..types import CategoryValue, CategoryType

class CategoryAnalyzer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def category_type(self) -> CategoryType:
        pass

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def analyze(
        self,
        query_text: str,
        query_id: Any,
        answer_ids: List[Any],
        skb: Any,
        adjacency_dict: Dict[int, List[int]],
        **kwargs
    ) -> CategoryValue:
        pass
```

**analyzers/registry.py - Mirror DescriptorRegistry pattern from `descriptors/registry.py`:**

```python
from typing import Dict, Type, List
import logging
from .base import CategoryAnalyzer

logger = logging.getLogger(__name__)

class CategoryAnalyzerRegistry:
    _analyzers: Dict[str, Type[CategoryAnalyzer]] = {}

    @classmethod
    def register(cls, analyzer_class: Type[CategoryAnalyzer] = None):
        def wrapper(klass: Type[CategoryAnalyzer]) -> Type[CategoryAnalyzer]:
            instance = klass()
            cls._analyzers[instance.name] = klass
            logger.debug(f"Registered category analyzer: {instance.name}")
            return klass
        if analyzer_class is not None:
            return wrapper(analyzer_class)
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[CategoryAnalyzer]:
        if name not in cls._analyzers:
            raise KeyError(f"Unknown analyzer: '{name}'. Available: {list(cls._analyzers.keys())}")
        return cls._analyzers[name]

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._analyzers.keys())
```

**analyzers/builtin.py - 5 Built-in Analyzers:**

1. `HopComplexityAnalyzer` - ORDINAL (1/2/3/4+) - Uses answer node graph depth
2. `EntityTypeAnalyzer` - DISCRETE - Regex patterns for dataset-specific entity types
3. `RetrievalDensityAnalyzer` - CONTINUOUS (0-1) - Pairwise similarity of answer embeddings
4. `QueryComplexityAnalyzer` - CONTINUOUS (0-1) - Linguistic features (length, constraints, negation)
5. `AnswerSetSizeAnalyzer` - DISCRETE (single/few/many) - Based on `len(answer_ids)`

### Phase 2: Pre-computation Script

**Create `stark/precompute_stark_categories.py`:**

Follow pattern from `stark/load_stark.py` lines 74-131 (`precompute_stark_adjacency`).

Key function:
```python
def precompute_stark_categories(
    dataset_name: str,
    analyzers: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    force_recompute: bool = False
) -> CategoryCache:
```

Cache location: `stark/category_cache/{dataset}_categories.pkl`

CLI:
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="prime", choices=["prime", "amazon", "mag"])
    parser.add_argument("--analyzers", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    # ...
```

### Phase 3: Multi-Archive Manager

**Create `swarm_rag_module/swarm_rag/evolution/map_elites/multi_archive.py`:**

```python
class MultiArchiveManager:
    def __init__(self, config: MultiArchiveConfig):
        self.archives: Dict[str, MapElitesArchive] = {}
        self.category_to_archive: Dict[str, str] = {}
        self.category_weights: Dict[str, float] = {}
        self.category_champions: Dict[str, Genome] = {}
        self.global_champion: Optional[Genome] = None
        self._initialize_archives()

    def add(self, genome: Genome, category: str) -> bool:
        """Add genome to category archive, update champions."""

    def add_with_routing(self, genome: Genome, category_fitness: Dict[str, float]) -> Dict[str, bool]:
        """Route genome to multiple archives based on per-category fitness."""

    def select_random(self, category: str = None) -> Optional[Genome]:
        """Select from specific archive or use selection_mode for cross-archive."""

    def select_parents_for_breeding(self) -> Tuple[Genome, Optional[Genome]]:
        """Respect cross_category_breeding config."""

    def get_champions(self) -> Dict[str, Genome]:
        """Return per-category champions + global."""

    def stats(self) -> MultiArchiveStats:
        """Aggregate stats across all archives."""
```

Selection modes: `uniform`, `weighted`, `round_robin`, `fitness_proportional`

### Phase 4: Category-Aware Evaluator

**Create `swarm_rag_module/swarm_rag/evolution/execution/category_evaluator.py`:**

Extend `PopulationEvaluator` to:
1. Accept `category_queries: Dict[str, List[Tuple[str, List[Any]]]]`
2. Track per-query category membership
3. Return `Dict[genome_id, Dict[category, fitness]]`

### Phase 5: Category Sampler

**Create `stark/category_sampler.py`:**

```python
class CategoryAwareSampler:
    def __init__(self, cache: CategoryCache):
        self.cache = cache

    def sample_stratified(self, n_samples: int, category_name: str, seed: int = None) -> List[int]:
        """Equal samples from each category value."""

    def sample_filtered(self, n_samples: int, filters: Dict[str, Any], seed: int = None) -> List[int]:
        """Sample matching specific category filters."""
```

### Phase 6: Multi-Archive Orchestrator

**Create `swarm_rag_module/swarm_rag/evolution/orchestrators/multi_archive_map_elites.py`:**

Template: `orchestrators/map_elites.py`

Key changes:
- Use `MultiArchiveManager` instead of single `MapElitesArchive`
- Use `CategoryAwareEvaluator`
- Route offspring to appropriate archives via `add_with_routing`
- Return `Dict[str, Genome]` (champions per category + global)

### Phase 7: Integration with evolve_stark.py

**Modify `stark/evolve_stark.py`:**

Add to config options:
```python
category_enabled: bool = True
category_name: str = "hop_complexity"
category_strategy: str = "per_category"  # or "stratified", "filtered"
```

Add `"category_aware"` preset to `stark/presets.yaml`:
```yaml
category_aware:
  n_generations: 30
  map_elites_enabled: true
  multi_archive_enabled: true
  category_name: "hop_complexity"
  selection_mode: "weighted"
  cross_category_breeding: true
  cross_breeding_rate: 0.3
```

### Phase 8: Config Extensions

**Modify `swarm_rag_module/swarm_rag/evolution/types/config.py`:**

Add:
```python
class CategoryConfig(TypedDict):
    name: str
    archive_key: str
    weight: float
    descriptors: List[str]
    bins: List[int]
    ranges: List[Tuple[float, float]]

class MultiArchiveConfigDict(TypedDict):
    multi_archive_enabled: bool
    categories: List[CategoryConfig]
    selection_mode: str
    cross_category_breeding: bool
    cross_breeding_rate: float
    champion_mode: str  # "per_category", "aggregate", "both"
```

---

## File Structure (Final)

```
stark/
  categories/
    __init__.py
    types.py
    analyzers/
      __init__.py
      base.py
      registry.py
      builtin.py
  category_cache/          # Created by precompute script
    prime_categories.pkl
    amazon_categories.pkl
    mag_categories.pkl
  precompute_stark_categories.py
  category_sampler.py
  load_stark.py            # Existing
  evolve_stark.py          # Modified
  presets.yaml             # Modified

swarm_rag_module/swarm_rag/evolution/
  map_elites/
    archive.py             # Existing
    loop.py                # Existing
    multi_archive.py       # NEW
    descriptors/           # Existing (pattern reference)
  execution/
    evaluator.py           # Existing
    category_evaluator.py  # NEW
  orchestrators/
    map_elites.py          # Existing
    multi_archive_map_elites.py  # NEW
  types/
    config.py              # Modified
```

---

## Verification Plan

1. **Unit tests:**
   - Each analyzer returns valid `CategoryValue`
   - `CategoryCache` serialization/deserialization
   - `MultiArchiveManager` add/select operations
   - Cross-category breeding rates

2. **Integration test:**
   ```bash
   python stark/precompute_stark_categories.py --dataset prime
   python stark/evolve_stark.py --dataset prime --preset category_aware
   ```

3. **Check outputs:**
   - Category cache files in `stark/category_cache/`
   - Per-category champions in evolution results
   - Multiple archives filling (coverage stats)
   - Different behavioral profiles per category

4. **Performance comparison:**
   - Standard MAP-Elites vs category-aware MAP-Elites
   - Verify specialists outperform generalists on their categories

---

## Implementation Order

1. Phase 1: Category types and registry (foundation)
2. Phase 2: Pre-computation script (enables testing analyzers)
3. Phase 3: Multi-archive manager (core new component)
4. Phase 5: Category sampler (simple, useful for testing)
5. Phase 4: Category-aware evaluator (depends on Phase 3)
6. Phase 6: Multi-archive orchestrator (depends on Phases 3-5)
7. Phase 8: Config extensions (can be done alongside Phase 6)
8. Phase 7: evolve_stark.py integration (final wiring)

---

## Notes

- The `DescriptorRegistry` pattern in `descriptors/registry.py` is clean and should be exactly mirrored for `CategoryAnalyzerRegistry`
- `precompute_stark_adjacency` in `load_stark.py:74-131` shows the exact caching pattern to follow
- Within-cell comparison in MAP-Elites uses `quality_score` - this stays the same
- The key change is that grid dimensions are now based on **question category** not **genome behavior**
