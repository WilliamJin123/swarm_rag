"""
RunManager - Unified storage management for evolution runs.

Manages checkpoints, logs, and results with device-aware save/load.
Supports async checkpointing for non-blocking I/O during evolution.
"""
import os
import json
import logging
from datetime import datetime
from threading import Thread, Event
from queue import Queue, Empty, Full
from typing import List, Optional, Any, TYPE_CHECKING
from dataclasses import asdict, is_dataclass

import torch

if TYPE_CHECKING:
    from ..types.config import StorageConfig, EvolutionConfig
    from ..types.genome import Genome

logger = logging.getLogger(__name__)


class AsyncCheckpointWriter:
    """
    Background thread for non-blocking checkpoint writes.

    Queues checkpoint data for async writing to disk, allowing the
    evolution loop to continue without waiting for I/O.

    Features:
    - Single-item queue (drops stale checkpoints if new one arrives)
    - Atomic update of latest.pkl via temp file
    - Automatic cleanup of old numbered checkpoints
    - Graceful shutdown with flush support
    """

    def __init__(self, keep_n_checkpoints: int = 10):
        """
        Initialize the async checkpoint writer.

        Args:
            keep_n_checkpoints: Number of numbered checkpoints to keep (0 = all)
        """
        self._queue: Queue = Queue(maxsize=1)
        self._shutdown = Event()
        self._keep_n = keep_n_checkpoints
        self._thread = Thread(target=self._writer_loop, daemon=True, name="AsyncCheckpointWriter")
        self._thread.start()

    def _writer_loop(self):
        """Background loop that processes checkpoint writes."""
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:  # Shutdown signal
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

    def _write_checkpoint(self, state: dict, gen_path: str, latest_path: str):
        """
        Perform the actual checkpoint write.

        Args:
            state: Checkpoint state dictionary
            gen_path: Path for numbered checkpoint (e.g., gen_050.pkl)
            latest_path: Path for latest.pkl
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(gen_path), exist_ok=True)

            # Save numbered checkpoint
            torch.save(state, gen_path)

            # Atomic update of latest.pkl
            temp = latest_path + ".tmp"
            torch.save(state, temp)
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.rename(temp, latest_path)

            logger.debug(f"Async checkpoint written: {gen_path}")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(os.path.dirname(gen_path))

        except Exception as e:
            logger.error(f"Failed to write checkpoint {gen_path}: {e}")

    def _cleanup_old_checkpoints(self, checkpoint_dir: str):
        """Remove checkpoints beyond keep_n limit."""
        if self._keep_n <= 0:
            return  # Keep all

        if not os.path.exists(checkpoint_dir):
            return

        # Find all numbered checkpoints
        try:
            ckpts = sorted(
                [f for f in os.listdir(checkpoint_dir)
                 if f.startswith("gen_") and f.endswith(".pkl")],
                reverse=True,  # Newest first
            )

            # Remove old ones
            for old in ckpts[self._keep_n:]:
                old_path = os.path.join(checkpoint_dir, old)
                try:
                    os.remove(old_path)
                    logger.debug(f"Removed old checkpoint: {old}")
                except OSError as e:
                    logger.warning(f"Failed to remove checkpoint {old}: {e}")
        except Exception as e:
            logger.warning(f"Checkpoint cleanup error: {e}")

    def save(self, state: dict, gen_path: str, latest_path: str):
        """
        Queue checkpoint for async write. Drops old pending if queue full.

        Args:
            state: Checkpoint state dictionary
            gen_path: Path for numbered checkpoint
            latest_path: Path for latest.pkl
        """
        try:
            self._queue.put_nowait((state, gen_path, latest_path))
        except Full:
            # Drop old pending checkpoint, queue new one
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait((state, gen_path, latest_path))
            except Full:
                logger.warning("Failed to queue checkpoint - writing synchronously")
                self._write_checkpoint(state, gen_path, latest_path)

    def flush(self):
        """Block until pending checkpoint is written."""
        self._queue.join()

    def shutdown(self):
        """Stop the background thread gracefully."""
        self._shutdown.set()
        try:
            self._queue.put_nowait(None)  # Wake up thread
        except Full:
            pass
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.warning("AsyncCheckpointWriter thread did not terminate cleanly")


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
            from ...utils.device import get_device
            device_str = get_device(force_cpu=(config.use_gpu == "never"))
            self.device = torch.device(device_str)
        else:
            self.device = device

        # Initialize async writer if enabled
        self._async_writer: Optional[AsyncCheckpointWriter] = None
        if getattr(config, 'async_checkpoints', True):
            self._async_writer = AsyncCheckpointWriter(
                keep_n_checkpoints=config.keep_n_checkpoints
            )
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
            # Async path: queue checkpoint and return immediately
            self._async_writer.save(state, gen_path, latest_path)
            logger.info(f"Checkpoint queued: {gen_path}")
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
        Flush pending checkpoints and shutdown async writer.

        Should be called before program exit to ensure all checkpoints are written.
        Safe to call multiple times or if async writer was not enabled.
        """
        if self._async_writer is not None:
            logger.info("Flushing pending checkpoints...")
            self._async_writer.flush()
            self._async_writer.shutdown()
            self._async_writer = None
            logger.info("Async checkpoint writer shut down")

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
        """Remove checkpoints beyond keep_n_checkpoints limit."""
        if self.config.keep_n_checkpoints <= 0:
            return  # Keep all

        if not os.path.exists(self.config.checkpoint_dir):
            return

        # Find all numbered checkpoints
        ckpts = sorted(
            [
                f for f in os.listdir(self.config.checkpoint_dir)
                if f.startswith("gen_") and f.endswith(".pkl")
            ],
            reverse=True,  # Newest first
        )

        # Remove old ones
        for old in ckpts[self.config.keep_n_checkpoints:]:
            old_path = os.path.join(self.config.checkpoint_dir, old)
            try:
                os.remove(old_path)
                logger.debug(f"Removed old checkpoint: {old}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {old}: {e}")

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


__all__ = ["RunManager", "AsyncCheckpointWriter"]
