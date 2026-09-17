"""Optional process-based parallelism (n_jobs).

Tasks are plain picklable payloads; workers rebuild the candidate library themselves,
so nothing that holds closures crosses the process boundary. On platforms that start
workers with "spawn" (macOS, Windows), call analyze(..., n_jobs=...) from code guarded by
`if __name__ == "__main__":`, as with any multiprocessing library.
"""
import os
from concurrent.futures import ProcessPoolExecutor


def resolve_jobs(n_jobs, n_tasks):
    if n_jobs is None or n_jobs == 1 or n_tasks <= 1:
        return 1
    cpus = os.cpu_count() or 1
    wanted = cpus + 1 + n_jobs if n_jobs < 0 else int(n_jobs)
    return max(1, min(wanted, n_tasks))


def pmap(func, payloads, workers):
    """Ordered map over payloads, in-process when workers == 1."""
    payloads = list(payloads)
    if workers <= 1:
        return [func(p) for p in payloads]
    chunk = max(1, len(payloads) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, payloads, chunksize=chunk))
