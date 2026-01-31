"""
RunManager - Unified storage management for evolution runs.

Manages checkpoints, logs, and results with device-aware save/load.
Supports async checkpointing for non-blocking I/O during evolution.
"""
import os
import json
import logging
import tempfile
import time
import copy
from datetime import datetime
from threading import Thread
from queue import Queue, Empty
from typing import List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict, is_dataclass

import torch

if TYPE_CHECKING:
    from ..types.config import StorageConfig, EvolutionConfig
    from ..types.genome import Genome

logger = logging.getLogger(__name__)


@dataclass
class CheckpointStats:
    """Statistics for checkpoint writing performance."""
    total_saves: int = 0
    total_time_seconds: float = 0.0
    total_bytes: int = 0


class AsyncCheckpointWriter:
    """
    Background thread for non-blocking checkpoint writes.

    Queues checkpoint data for async writing to disk, allowing the
    evolution loop to continue without waiting for I/O.

    Features:
    - Unbounded queue (all checkpoints are written, none dropped)
    - Non-daemon thread (waits for queue drain on shutdown)
    - Atomic writes via temp file + os.replace
    - Retry logic with cleanup on failure
    - Comprehensive logging with stats tracking
    """

    def __init__(self):
        """Initialize the async checkpoint writer."""
        self._queue: Queue = Queue()  # Unbounded queue - queue all checkpoints
        self._stats = CheckpointStats()
        self._thread = Thread(
            target=self._writer_loop,
            daemon=False,  # Non-daemon: allow graceful shutdown
            name="AsyncCheckpointWriter"
        )
        self._thread.start()

    def _prepare_checkpoint_data(self, state: dict) -> dict:
        """
        Deep copy checkpoint data for safe queuing to background thread.

        Handles:
        - Tensors: detach, clone, move to CPU
        - Genomes: use .copy() if available
        - Nested dicts/lists: recursive deep copy

        Args:
            state: Checkpoint state dictionary

        Returns:
            Deep-copied state safe for background thread
        """
        return self._deep_copy_state(state)

    def _deep_copy_state(self, obj: Any) -> Any:
        """Recursively deep copy state, handling tensors and genomes."""
        if isinstance(obj, torch.Tensor):
            # Detach from computation graph, clone, move to CPU
            # Use .data to ensure requires_grad=False on the result
            return obj.detach().clone().cpu().requires_grad_(False)
        elif isinstance(obj, dict):
            # Handle dicts before generic .copy() check (dicts have .copy())
            return {k: self._deep_copy_state(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy_state(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._deep_copy_state(item) for item in obj)
        elif hasattr(obj, 'copy') and callable(obj.copy):
            # Genome or similar objects with copy method
            return obj.copy()
        else:
            # Primitive types or immutable objects - use copy.deepcopy for safety
            try:
                return copy.deepcopy(obj)
            except Exception:
                # If deepcopy fails, return as-is (primitives, etc.)
                return obj

    def _writer_loop(self):
        """Background loop that processes checkpoint writes."""
        while True:
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:  # Sentinel: shutdown signal
                    self._queue.task_done()
                    break
                state, gen_path, latest_path = item
                self._write_checkpoint(state, gen_path, latest_path)
                self._queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"AsyncCheckpointWriter error: {e}")
                try:
                    self._queue.task_done()
                except ValueError:
                    pass  # Already marked done

    def _write_checkpoint(self, state: dict, gen_path: str, latest_path: str) -> bool:
        """
        Perform atomic checkpoint write with retry logic.

        Uses temp file + os.replace for atomic writes.
        Retries once on failure before logging error.

        Args:
            state: Checkpoint state dictionary
            gen_path: Path for numbered checkpoint (e.g., gen_050.pkl)
            latest_path: Path for latest.pkl

        Returns:
            True if successful, False if failed after retries
        """
        start_time = time.time()
        max_attempts = 2
        checkpoint_dir = os.path.dirname(gen_path)

        # Ensure directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        generation = state.get('generation', '?')

        for attempt in range(1, max_attempts + 1):
            temp_gen_path = None
            temp_latest_path = None
            try:
                # Write numbered checkpoint atomically
                fd_gen, temp_gen_path = tempfile.mkstemp(
                    suffix=".tmp", prefix="ckpt_", dir=checkpoint_dir
                )
                os.close(fd_gen)  # Close fd before torch.save
                torch.save(state, temp_gen_path)
                os.replace(temp_gen_path, gen_path)
                temp_gen_path = None  # Successfully moved, don't cleanup

                # Write latest checkpoint atomically
                fd_latest, temp_latest_path = tempfile.mkstemp(
                    suffix=".tmp", prefix="ckpt_", dir=checkpoint_dir
                )
                os.close(fd_latest)
                torch.save(state, temp_latest_path)
                os.replace(temp_latest_path, latest_path)
                temp_latest_path = None  # Successfully moved, don't cleanup

                # Calculate stats
                elapsed = time.time() - start_time
                file_size = os.path.getsize(gen_path)

                # Update cumulative stats
                self._stats.total_saves += 1
                self._stats.total_time_seconds += elapsed
                self._stats.total_bytes += file_size

                # Log completion
                size_mb = file_size / (1024 * 1024)
                logger.info(f"Checkpoint saved (gen {generation}, {size_mb:.1f}MB, {elapsed:.2f}s)")

                return True

            except Exception as e:
                # Clean up temp files on failure
                for temp_path in [temp_gen_path, temp_latest_path]:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

                if attempt < max_attempts:
                    logger.warning(
                        f"Checkpoint write failed (attempt {attempt}), retrying... Error: {e}"
                    )
                    time.sleep(0.5)
                else:
                    logger.error(
                        f"Checkpoint write failed after {max_attempts} attempts: {e}"
                    )
                    return False

        return False

    def queue_checkpoint(self, state: dict, gen_path: str, latest_path: str):
        """
        Queue checkpoint for async write.

        Args:
            state: Pre-copied checkpoint state dictionary
            gen_path: Path for numbered checkpoint
            latest_path: Path for latest.pkl
        """
        generation = state.get('generation', '?')

        self._queue.put((state, gen_path, latest_path))
        logger.info(f"Checkpoint queued (gen {generation})")

        # Warn if queue is backing up
        queue_size = self._queue.qsize()
        if queue_size > 1:
            logger.warning(f"Checkpoint queue: {queue_size} pending")

    def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Stop the background thread gracefully, waiting for queue to drain.

        Args:
            timeout: Maximum seconds to wait for queue drain

        Returns:
            True if clean shutdown, False if timeout
        """
        # Send sentinel to signal shutdown
        self._queue.put(None)

        # Wait for thread to finish
        self._thread.join(timeout=timeout)

        # Log summary stats
        if self._stats.total_saves > 0:
            avg_time = self._stats.total_time_seconds / self._stats.total_saves
            logger.info(
                f"Checkpointing: {self._stats.total_saves} saves, "
                f"{self._stats.total_time_seconds:.1f}s total, avg {avg_time:.2f}s"
            )

        if self._thread.is_alive():
            logger.warning("AsyncCheckpointWriter thread did not terminate cleanly")
            return False

        return True


class RunManager:
    """
    Manages evolution run storage - checkpoints, logs, results.

    Device-aware save/load:
    - Saves tensors with device info preserved
    - Loads tensors directly to current device (no intermediate transfers)
    - Handles CPU<->GPU transitions automatically via map_location

    Async checkpointing:
    - When enabled, checkpoint writes happen in background thread
    - Evolution loop continues without waiting for I/O
    - Call close() to flush pending checkpoints before exit
    """

    def __init__(self, config: "StorageConfig", device: torch.device = None):
        """
        Initialize RunManager.

        Args:
            config: StorageConfig with paths and settings
            device: Target device for tensor operations (auto-detected if None)
        """
        self.config = config

        # Auto-detect device if not specified
        if device is None:
            from ...utils.device import resolve_device
            device_str = resolve_device(config.device)
            self.device = torch.device(device_str)
        else:
            self.device = device

        # Initialize async writer if enabled
        self._async_writer: Optional[AsyncCheckpointWriter] = None
        if getattr(config, 'async_checkpoints', True):
            self._async_writer = AsyncCheckpointWriter()
            logger.info("Async checkpoint writing enabled")

    def initialize_run(self, full_config: "EvolutionConfig" = None):
        """
        Create directories and save config snapshot.

        Args:
            full_config: Full evolution config to snapshot for reproducibility
        """
        self.config.ensure_directories()
        logger.info(f"Initialized run directory: {self.config.run_dir}")

        if full_config:
            self._save_config_snapshot(full_config)

    def save_checkpoint(
        self,
        population: List["Genome"],
        best_genome: "Genome",
        generation: int,
        extra_state: dict = None,
        random_state: Any = None,
        torch_rng_state: Any = None,
        tracker_history: Any = None,
    ):
        """
        Save numbered checkpoint + update latest.pkl atomically.

        Uses torch.save for device-aware serialization.
        When async_checkpoints is enabled, writes happen in background thread.

        Args:
            population: Current population (archive contents)
            best_genome: Best genome found so far
            generation: Current generation number
            extra_state: Additional state (e.g., archive metadata)
            random_state: Python random state for reproducibility
            torch_rng_state: PyTorch random state for reproducibility
            tracker_history: Progress tracker history
        """
        state = {
            "generation": generation,
            "population": population,
            "best_genome": best_genome,
            "timestamp": datetime.now().isoformat(),
            "dataset": self.config.dataset,
            "run_id": self.config.run_id,
            "device": str(self.device),
        }

        if random_state is not None:
            state["random_state"] = random_state
        if torch_rng_state is not None:
            state["torch_rng_state"] = torch_rng_state
        if tracker_history is not None:
            state["tracker_history"] = tracker_history
        if extra_state is not None:
            state.update(extra_state)

        gen_path = self.config.checkpoint_path_for_gen(generation)
        latest_path = self.config.latest_checkpoint_path

        if self._async_writer is not None:
            # Async path: deep copy state and queue for background write
            prepared_state = self._async_writer._prepare_checkpoint_data(state)
            self._async_writer.queue_checkpoint(prepared_state, gen_path, latest_path)
        else:
            # Synchronous path: write immediately
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            torch.save(state, gen_path)

            # Atomic update of latest
            temp = latest_path + ".tmp"
            torch.save(state, temp)
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.rename(temp, latest_path)

            logger.info(f"Checkpoint saved: {gen_path}")
            self._cleanup_old_checkpoints()

    def load_checkpoint(self, path: str = None) -> dict:
        """
        Load checkpoint, moving tensors to current device.

        Handles GPU<->CPU transitions automatically via map_location.

        Args:
            path: Specific checkpoint path (uses latest.pkl if None)

        Returns:
            Checkpoint state dictionary with tensors on current device
        """
        if path is None:
            path = self.config.latest_checkpoint_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load and map tensors to current device
        state = torch.load(path, map_location=self.device, weights_only=False)
        logger.info(f"Loaded checkpoint from {path} (device: {self.device})")
        return state

    def save_best_genome(self, genome: "Genome"):
        """
        Save best genome as JSON.

        Args:
            genome: Best genome to save
        """
        os.makedirs(self.config.results_dir, exist_ok=True)
        with open(self.config.best_genome_path, "w") as f:
            json.dump(genome.to_dict(), f, indent=2)
        logger.info(f"Best genome saved: {self.config.best_genome_path}")

    def save_final_metrics(self, metrics: dict):
        """
        Save final evaluation metrics.

        Args:
            metrics: Dictionary of final metrics
        """
        os.makedirs(self.config.results_dir, exist_ok=True)
        path = os.path.join(self.config.results_dir, "final_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Final metrics saved: {path}")

    def close(self):
        """
        Shutdown async writer, waiting for queue to drain.

        Should be called before program exit to ensure all checkpoints are written.
        Safe to call multiple times or if async writer was not enabled.
        """
        if self._async_writer is not None:
            logger.info("Shutting down async checkpoint writer...")
            clean = self._async_writer.shutdown(timeout=30.0)
            if clean:
                logger.info("Async checkpoint writer shut down cleanly")
            else:
                logger.warning("Async checkpoint writer shutdown timed out")
            self._async_writer = None

    def _save_config_snapshot(self, config: "EvolutionConfig"):
        """
        Save full config for reproducibility.

        Args:
            config: Full evolution config to snapshot
        """
        config_dict = self._to_dict(config)
        with open(self.config.config_snapshot_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Config snapshot saved: {self.config.config_snapshot_path}")

    def _cleanup_old_checkpoints(self):
        """
        Checkpoint rotation disabled - all checkpoints kept per CONTEXT.md.

        This method is kept for API compatibility with synchronous path
        but performs no cleanup.
        """
        # Checkpoint rotation disabled - all checkpoints kept per CONTEXT.md
        return

    @staticmethod
    def _to_dict(obj: Any) -> Any:
        """
        Recursively convert dataclass/object to dict for JSON serialization.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable representation
        """
        if is_dataclass(obj) and not isinstance(obj, type):
            # Check if dataclass has its own to_dict method (like WeightTensors)
            if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                result[field_name] = RunManager._to_dict(value)
            return result
        elif isinstance(obj, dict):
            return {k: RunManager._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [RunManager._to_dict(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

    @staticmethod
    def list_runs(base_dir: str, dataset: str = None) -> List[dict]:
        """
        List all runs, optionally filtered by dataset.

        Args:
            base_dir: Base runs directory
            dataset: Filter by dataset name (optional)

        Returns:
            List of run info dicts: {"dataset": str, "run_id": str, "path": str}
        """
        runs = []
        if not os.path.exists(base_dir):
            return runs

        datasets = [dataset] if dataset else os.listdir(base_dir)

        for ds in datasets:
            ds_path = os.path.join(base_dir, ds)
            if not os.path.isdir(ds_path):
                continue

            for run_id in os.listdir(ds_path):
                run_path = os.path.join(ds_path, run_id)
                if os.path.isdir(run_path):
                    runs.append({
                        "dataset": ds,
                        "run_id": run_id,
                        "path": run_path,
                    })

        # Sort by run_id (timestamp-based) descending (newest first)
        return sorted(runs, key=lambda x: x["run_id"], reverse=True)

    @staticmethod
    def find_latest_run(base_dir: str, dataset: str) -> Optional[str]:
        """
        Find most recent run for a dataset.

        Args:
            base_dir: Base runs directory
            dataset: Dataset name

        Returns:
            Path to most recent run directory, or None if no runs exist
        """
        runs = RunManager.list_runs(base_dir, dataset)
        return runs[0]["path"] if runs else None


__all__ = ["RunManager", "AsyncCheckpointWriter", "CheckpointStats"]
