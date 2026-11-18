import logging
import time
from contextlib import contextmanager
from typing import Any, Dict

logger = logging.getLogger("imci_profiler")


class PipelineProfiler:
    """
    High-precision performance profiler for the IMCI recommendation pipeline.
    Measures and logs execution times of each stage using time.perf_counter().
    """

    def __init__(self, operation_name: str = "Recommendation Pipeline"):
        self.operation_name = operation_name
        self.stages: Dict[str, float] = {}
        self._stage_starts: Dict[str, float] = {}
        self.overall_start = time.perf_counter()
        self.overall_end: float = 0.0

    @contextmanager
    def measure(self, stage_name: str):
        """Context manager to measure a pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages[stage_name] = round(elapsed, 4)
            logger.info("Stage '%s' completed in %.4f s (%.2f ms)", stage_name, elapsed, elapsed * 1000)

    def start_stage(self, stage_name: str) -> None:
        self._stage_starts[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        start = self._stage_starts.pop(stage_name, None)
        if start is not None:
            elapsed = time.perf_counter() - start
            self.stages[stage_name] = round(elapsed, 4)
            logger.info("Stage '%s' completed in %.4f s (%.2f ms)", stage_name, elapsed, elapsed * 1000)
            return elapsed
        return 0.0

    def finish(self) -> Dict[str, Any]:
        """Finalize total runtime and compile formatted report."""
        self.overall_end = time.perf_counter()
        total_seconds = round(self.overall_end - self.overall_start, 4)

        formatted_stages = {}
        for stage, duration in self.stages.items():
            if duration < 1.0:
                formatted_stages[stage] = f"{duration * 1000:.2f} ms"
            else:
                formatted_stages[stage] = f"{duration:.2f} s"

        logger.info(
            "[%s] Finished in %.4f s. Stage breakdown: %s",
            self.operation_name,
            total_seconds,
            formatted_stages,
        )

        return {
            "total_duration_seconds": total_seconds,
            "stage_durations_seconds": self.stages,
            "formatted_stages": formatted_stages,
        }
