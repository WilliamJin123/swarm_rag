import os
import pickle
import uuid
import glob
import random
from typing import List
from .base import EvolutionExtension
from ..types.genome import Genome

class FileMigrationExtension(EvolutionExtension):
    def __init__(
        self, 
        migration_dir: str = "./migration_pool", 
        interval: int = 5, 
        rate: float = 0.1,
        island_id: str = None
    ):
        """
        Asynchronous Island Model via File System.
        
        Args:
            migration_dir: Shared folder where all islands drop their migrants.
            interval: How many generations between migrations.
            rate: Percentage of population to swap (0.1 = 10%).
            island_id: Unique name for this engine. Defaults to random UUID.
        """
        self.migration_dir = migration_dir
        self.interval = interval
        self.rate = rate
        self.island_id = island_id or str(uuid.uuid4())[:8]
        
        # Ensure the shared pool exists
        os.makedirs(self.migration_dir, exist_ok=True)
        
        # Track which migration files we have already processed
        self.processed_files = set()

    def on_generation_end(self, ctx):
        """
        Called at end of generation. Handles Export and Import.
        """
        if ctx.generation == 0: return
        if ctx.generation % self.interval != 0: return

        print(f"  [Island {self.island_id}] Triggering Migration...")
        
        # 1. EXPORT: Save Top K Genomes
        self._export_migrants(ctx)
        
        # 2. IMPORT: Load Fresh Migrants from neighbors
        self._import_migrants(ctx)

    def _export_migrants(self, ctx):
        # Sort Best -> Worst
        ctx.population.sort(key=lambda g: g.fitness.quality_score, reverse=True)
        
        # Take top K%
        count = int(len(ctx.population) * self.rate)
        migrants = ctx.population[:count]
        
        # Save to file: "migration_pool/gen_10_island_A1B2.pkl"
        filename = f"gen_{ctx.generation}_island_{self.island_id}.pkl"
        path = os.path.join(self.migration_dir, filename)
        
        # We save copies to avoid mutation issues
        payload = [g.copy() for g in migrants]
        
        with open(path, "wb") as f:
            pickle.dump(payload, f)
            
        print(f"    -> Exported {len(migrants)} genomes to {filename}")
        
        # Mark our own file as processed so we don't re-import it
        self.processed_files.add(filename)

    def _import_migrants(self, ctx):
        # Look for ALL pickle files in the pool
        pattern = os.path.join(self.migration_dir, "gen_*.pkl")
        candidates = glob.glob(pattern)
        
        new_blood: List[Genome] = []
        
        for path in candidates:
            filename = os.path.basename(path)
            
            # Skip files we already processed OR files we created ourselves
            if filename in self.processed_files:
                continue
            if f"island_{self.island_id}" in filename:
                continue
                
            try:
                with open(path, "rb") as f:
                    incoming = pickle.load(f)
                    if isinstance(incoming, list):
                        new_blood.extend(incoming)
                        print(f"    <- Imported {len(incoming)} genomes from {filename}")
                
                # Mark as seen so we don't import duplicates next time
                self.processed_files.add(filename)
                
            except Exception as e:
                print(f"    ! Failed to load migrant file {filename}: {e}")

        # If we found new genomes, inject them
        if new_blood:
            self._inject_migrants(ctx, new_blood)

    def _inject_migrants(self, ctx, migrants: List[Genome]):
        """Replaces the worst members of the population with immigrants."""
        pop = ctx.population

        # Sort current population: Best -> Worst
        pop.sort(key=lambda g: g.fitness.quality_score, reverse=True)
        
        num_replace = len(migrants)
        total_pop = len(pop)
        
        # Don't replace more than 50% of the population to maintain stability
        limit = int(total_pop * 0.5)
        if num_replace > limit:
            # Shuffle and take random subset if too many incomers
            random.shuffle(migrants)
            migrants = migrants[:limit]
            num_replace = limit
            
        # Replace the WORST individuals (at the end of the list)
        # Note: We assume migrants are 'Elites' from other islands, so generally good.
        ctx.population[-num_replace:] = migrants
        print(f"    => Integrated {num_replace} immigrants into population.")