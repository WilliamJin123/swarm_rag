# MAP-Elites Integration Plan

## Overview
This document outlines the plan to integrate the **Multi-dimensional Archive of Phenotypic Elites (MAP-Elites)** algorithm into the `swarm_rag` evolution framework. MAP-Elites differs from standard evolutionary algorithms by maintaining a structured archive of high-performing solutions spread across a user-defined behavioral feature space, rather than a single ranked population. This encourages diversity and illumination of the search space.

## Architecture

### 1. New Components

#### `MapElitesArchive`
*   **Responsibility**: Manages the grid/map of elites.
*   **Data Structure**: A dictionary or N-dimensional array mapping `(f1_bin, f2_bin, ...)` tuples to `Genome` objects.
*   **Key Methods**:
    *   `add(genome, descriptor) -> bool`: Attempts to add a genome. Replaces existing if fitness is higher. Returns `True` if added/replaced.
    *   `select_random() -> Genome`: Returns a random elite for reproduction.
    *   `as_population() -> List[Genome]`: Returns all elites as a list (for compatibility).
    *   `stats()`: Returns coverage (%), max fitness, etc.

#### `DescriptorCalculator`
*   **Responsibility**: Extracts behavioral features (descriptors) from a Genome.
*   **Types of Descriptors**:
    *   **Genotypic**: Calculated directly from genome parameters (e.g., `complexity`, `mutation_rate`, `decay_param`).
    *   **Phenotypic**: Calculated from evaluation metrics (e.g., `latency`, `avg_steps`, `cost`). *Note: Requires evaluation before placement.*

#### `MapElitesLoop`
*   **Responsibility**: The execution logic specific to MAP-Elites.
*   **Logic**:
    1.  Select `k` parents from the `Archive`.
    2.  Apply Mutation/Crossover (standard strategies).
    3.  (Engine evaluates offspring).
    4.  Calculate descriptors for offspring.
    5.  Add offspring to `Archive`.

### 2. Integration with `EvolutionEngine`

The `EvolutionEngine` currently assumes a list-based population. We will adapt it to support a "MAP-Elites Mode":

*   **Initialization**: If configured for MAP-Elites, initialize `MapElitesArchive`.
*   **Generation Loop**:
    *   Instead of `self.loop.step(population)`, call `self.map_elites_loop.step(archive)`.
    *   The "Population" effectively becomes the batch of offspring being evaluated in the current generation + the Archive contents for reporting.
*   **Logging**: Update `ProgressTracker` to log archive coverage and QD-score (sum of fitness of all elites) in addition to best fitness.

## File Structure

New directory: `swarm_rag_module/swarm_rag/evolution/map_elites/`

*   `__init__.py`
*   `archive.py`: `MapElitesArchive` class.
*   `descriptors.py`: `DescriptorCalculator` and concrete implementations.
*   `loop.py`: `MapElitesLoop` class.

## Configuration Updates

Add to `EvolutionConfigDict` in `types/config.py`:

```python
# MAP-Elites Configuration
map_elites_enabled: bool = False
map_elites_dims: List[str] = ["complexity", "cost"] # Names of descriptors
map_elites_bins: List[int] = [10, 10] # Number of bins per dimension
map_elites_ranges: List[Tuple[float, float]] = [(0, 100), (0, 1.0)] # Min/Max for each dim
map_elites_initial_fill: int = 100 # Initial random population to seed archive
```

## Step-by-Step Implementation Plan

### Phase 1: Core Components
1.  **Define Configuration**: Update `EvolutionConfigDict` with MAP-Elites parameters.
2.  **Implement `DescriptorCalculator`**: Create the logic to extract features. Start with simple Genotypic features (e.g., `complexity` from tree size, `n_agents` from params).
3.  **Implement `MapElitesArchive`**: Create the grid structure and addition logic.

### Phase 2: Logic & Loop
1.  **Implement `MapElitesLoop`**: Create the loop that selects from archive and produces offspring.
2.  **Strategies**: Ensure existing Mutation/Crossover strategies work seamlessly (they should, as they operate on Genomes).

### Phase 3: Engine Integration
1.  **Modify `EvolutionEngine`**:
    *   Add conditional logic in `__init__` to setup MAP-Elites components if enabled.
    *   Add conditional logic in `optimize` loop to use `MapElitesLoop`.
    *   Handle the flow: `Select -> Breed -> Evaluate -> Compute Descriptors -> Add to Archive`.

### Phase 4: Logging & Verification
1.  **Update Tracker**: Log QD metrics (Coverage, QD-Score).
2.  **Testing**: Create a test script `tests/integration/test_map_elites.py` to verify the archive fills up and improves over time.

## Feature Descriptors (Candidates)

1.  **Complexity**: Size of strategy trees. (Genotypic)
2.  **Cost**: `cost_score` from fitness (Phenotypic - requires eval).
3.  **Aggressiveness**: `n_agents` * `steps` (Genotypic).
4.  **Exploration**: `decay` parameter or specific tree features (Genotypic).
5.  **Latency**: Execution time (Phenotypic).

*Recommendation*: Start with **Complexity** (Genotypic) and **Aggressiveness** (Genotypic) for easier debugging, then move to Phenotypic.
