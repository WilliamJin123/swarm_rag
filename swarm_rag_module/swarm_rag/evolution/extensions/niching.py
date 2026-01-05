from typing import Dict, List, Set
import numpy as np
import random
from scipy.spatial.distance import pdist, squareform

from ...core.heuristics import HeuristicRegistry
from .base import EvolutionExtension
from ..types.genome import Genome

class NichingExtension(EvolutionExtension):
    def __init__(self, sigma_share: float = 2.0, alpha: float = 1.0, n_probes: int = 10):
        """
        fitness sharing (Niching) extension using Behavioral Profiling.
        
        Args:
            sigma_share: Niche radius. Since we normalize vectors, 1.0-2.0 is a good default.
            alpha: Shape of the sharing function (1.0 is linear).
            n_probes: Number of random test cases to run on each genome's tree.
        """
        self.sigma_share = sigma_share
        self.alpha = alpha
        self.n_probes = n_probes
        self.feature_keys = [getattr(k, 'value', k) for k in HeuristicRegistry.all().keys()]
        self.probes = self._generate_probes(n_probes)

    def _generate_probes(self, n: int) -> List[Dict[str, float]]:
        """Creates N random scenarios using feature names."""
        probes = []
        for _ in range(n):
            probe = {}
            for key in self.feature_keys:
                # Heuristics generally operate on normalized scores (0-1) 
                # or counts/degrees (0-50+). 
                # A range of 0.0 to 10.0 covers enough variance to differentiate behaviors.
                probe[key] = random.uniform(0.0, 10.0)
            probes.append(probe)
        return probes

    def on_after_evaluation(self, ctx):
        """
        Adjusts fitness scores based on crowding.
        """
        pop = ctx.population
        n_pop = len(pop)
        if n_pop < 2: return

        discovered_keys: Set[str] = set()
        for g in pop:
            # Check compiled cache (preferred) and raw strategies
            discovered_keys.update(g._compiled_cache.keys())
            discovered_keys.update(g.strategies.keys())

        target_keys = sorted(list(discovered_keys))

        signatures = []
        for g in pop:
            genome_signature = []
            
            for p_key in sorted(g.params.keys()):
                val = g.params[p_key]
                if isinstance(val, (int, float)):
                    genome_signature.append(float(val))

            for strat_key in target_keys:
                func = g._compiled_cache.get(strat_key)
                
                if func is None:
                    tree = g.strategies.get(strat_key)
                    if not tree:
                        print(f"{strat_key} strategy missing, zero-filling")
                        genome_signature.extend([0.0] * self.n_probes)
                        continue
                    func = tree.evaluate
            
                try:
                    # Run the function on all probes
                    vals = [float(func(p)) for p in self.probes]
                    # Clamp
                    vals = [max(-100.0, min(100.0, x)) for x in vals]
                    genome_signature.extend(vals)
                except Exception:
                    genome_signature.extend([0.0] * self.n_probes)  
            
            signatures.append(genome_signature)

        signatures = np.array(signatures)

        # Safety: If signature is empty (no params, no strategies), do nothing
        if signatures.shape[1] == 0:
            return

        # Normalize
        std_dev = signatures.std(axis=0)
        std_dev[std_dev == 0] = 1.0 
        signatures = (signatures - signatures.mean(axis=0)) / std_dev

        dists = squareform(pdist(signatures, metric='euclidean'))

        # Fitness sharing
        for i in range(n_pop):
            niche_count = 0.0
            for j in range(n_pop):
                d = dists[i][j]
                if d < self.sigma_share:
                    niche_count += (1.0 - (d / self.sigma_share)) ** self.alpha
            
            pop[i].fitness.quality_score /= max(1.0, niche_count)