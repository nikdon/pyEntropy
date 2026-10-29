"""Run benchmark cases for pyentrp.

    python benchmarks/bench.py                         # every measure, Markdown table
    python benchmarks/bench.py --only sample fuzzy     # subset, matched on the label
    python benchmarks/bench.py --sizes 1000 5000 --repeats 5
    python benchmarks/bench.py --json results.json     # also save raw timings
    python benchmarks/bench.py --no-isolate            # run everything in this process (faster, for debugging)

Each case runs in its own fresh Python process (single-threaded BLAS), so one case cannot affect another through
memory layout, caches or leftover allocations. Timing uses `timeit`. Case definitions and shared inputs live in
`benchmarks/cases.py`.
"""

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import timeit
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np

from pyentrp import entropy as ent

if __package__:
    from .cases import CASES, BenchmarkData, Case, make_data
else:
    from cases import CASES, BenchmarkData, Case, make_data

RESULT_MARKER = "BENCH_RESULT "  # prefix of the JSON line a worker process prints
SLOW_CALL_SECONDS = 2.0


def time_case(case: Case, data: BenchmarkData, repeats: int, slow: float = SLOW_CALL_SECONDS) -> float:
    """Seconds per call: the best of `repeats` timeit runs.

    `timeit` picks the loop count (autorange) so that fast calls are measured over >= 0.2 s, and keeps garbage
    collection out of the timed region. A call that is already slow is timed once and not repeated.
    """
    timer = timeit.Timer(lambda: case.run(data))
    number, total = timer.autorange()
    if total / number > slow:
        return total / number
    return min(timer.repeat(repeat=repeats, number=number)) / number


def check_finite(case: Case, result: object) -> None:
    """Reject non-finite results from a benchmark function."""
    if not np.all(np.isfinite(np.asarray(result, dtype=float))):
        raise RuntimeError(f"{case.label!r} returned non-finite values: {result!r}")


def measure(case: Case, sizes: list[int], repeats: int, budget: float) -> dict[int, float | None]:
    """Measure one case at each requested size.

    Skip a size if its projected runtime, assuming quadratic growth, exceeds `budget` seconds.
    """
    row: dict[int, float | None] = {}
    prev: tuple[int, float] | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_finite(case, case.run(make_data(sizes[0])))  # warm-up, also validates the result
        for n in sizes:
            if prev and prev[1] * (n / prev[0]) ** 2 > budget:
                row[n] = None
                continue
            t = time_case(case, make_data(n), repeats)
            row[n] = t
            prev = (n, t)
    return row


def measure_isolated(case: Case, sizes: list[int], repeats: int, budget: float) -> dict[int, float | None]:
    """Measure one case in a fresh interpreter, isolated from other cases."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        case.label,
        "--sizes",
        *map(str, sizes),
        "--repeats",
        str(repeats),
        "--budget",
        str(budget),
    ]
    env = {**os.environ, **dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1")}
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark {case.label!r} failed:\n{proc.stderr.strip() or proc.stdout.strip()}")
    line = next(ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(RESULT_MARKER))
    return {int(n): t for n, t in json.loads(line[len(RESULT_MARKER) :]).items()}


def run_all(
    cases: list[Case], sizes: list[int], repeats: int, budget: float, isolate: bool = True
) -> dict[str, dict[int, float | None]]:
    """Run selected benchmark cases and return their timings."""
    run = measure_isolated if isolate else measure
    results = {}
    for case in cases:
        results[case.label] = run(case, sizes, repeats, budget)
        print(f"  done: {case.label}", file=sys.stderr, flush=True)
    return results


def fmt(t: float | None) -> str:
    """Format a timing in seconds as a compact string with units."""
    if t is None:
        return "-"
    if t < 1e-3:
        return f"{t * 1e6:.0f} µs"
    return f"{t * 1e3:.1f} ms" if t < 1 else f"{t:.1f} s"


def cpu_name() -> str:
    """Human-readable CPU model, e.g. 'Apple M3 Pro' or 'Intel(R) Core(TM) i7-1185G7'."""
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def provenance_warnings() -> list[str]:
    """Catch the two ways a benchmark can silently measure something other than the checkout it sits in."""
    root = Path(__file__).resolve().parent.parent
    code = Path(ent.__file__).resolve()
    problems = []
    if (root / "pyentrp").is_dir() and root not in code.parents:
        problems.append(f"timing the installed pyentrp at {code.parent}, not the checkout at {root}")
    try:
        declared = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    except OSError:
        declared = None
    installed = metadata.version("pyentrp")
    if declared and declared.group(1) != installed:
        problems.append(
            f"installed metadata says {installed} but pyproject.toml says {declared.group(1)}; "
            "reinstall (e.g. pip install -e .) so the reported version is right"
        )
    return problems


def environment() -> dict[str, str]:
    """Collect package, runtime and CPU details for benchmark output."""
    return {
        "pyentrp": metadata.version("pyentrp"),
        "pyentrp_path": str(Path(ent.__file__).resolve().parent),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": cpu_name(),
        "logical_cpus": str(os.cpu_count()),  # installed on the machine, not used by the benchmark
    }


def markdown(
    results: dict[str, dict[int, float | None]], sizes: list[int], budget: float, repeats: int, isolate: bool
) -> str:
    """Render benchmark results and environment details as Markdown."""
    env = environment()
    lines = [
        (
            f"pyentrp {env['pyentrp']}, Python {env['python']}, NumPy {env['numpy']}, {env['cpu']}. "
            f"White noise, best of up to {repeats} timeit repeats"
            f" (calls over {SLOW_CALL_SECONDS:g} s are timed once)"
            f"{', each benchmark in its own single-threaded process' if isolate else ''}."
        ),
        "",
        "| Function | " + " | ".join(f"N = {n:,}" for n in sizes) + " |",
        "|---|" + "---|" * len(sizes),
    ]
    for label, row in results.items():
        lines.append(f"| `{label}` | " + " | ".join(fmt(row[n]) for n in sizes) + " |")
    if any(t is None for row in results.values() for t in row.values()):
        lines += ["", f"`-` = skipped: projected to take longer than {budget:g} s."]
    return "\n".join(lines)


def main() -> int:
    """Parse command-line options, run benchmarks and print their results."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000], help="series lengths")
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help=f"timeit repeats per size; calls over {SLOW_CALL_SECONDS:g} s are timed once",
    )
    p.add_argument("--budget", type=float, default=120.0, help="skip a size projected to take longer than this (s)")
    p.add_argument("--only", nargs="+", metavar="TEXT", help="run only cases whose label contains any of these")
    p.add_argument("--json", metavar="PATH", help="also write raw timings to this file")
    p.add_argument("--no-isolate", action="store_true", help="run all cases in this process")
    p.add_argument("--worker", metavar="LABEL", help=argparse.SUPPRESS)  # internal: run one case, print JSON
    args = p.parse_args()
    sizes = sorted(set(args.sizes))

    if any(size < 1 for size in sizes):
        p.error("--sizes values must be positive integers")
    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    if not math.isfinite(args.budget) or args.budget <= 0:
        p.error("--budget must be a finite positive number")

    if args.worker:
        case = next(c for c in CASES if c.label == args.worker)
        row = measure(case, sizes, args.repeats, args.budget)
        print(RESULT_MARKER + json.dumps(row))
        return 0

    for problem in provenance_warnings():
        print(f"WARNING: {problem}", file=sys.stderr)
    cases = CASES
    if args.only:
        needles = [s.lower() for s in args.only]
        cases = [c for c in CASES if any(s in c.label.lower() for s in needles)]
        if not cases:
            print("No benchmark matches --only", file=sys.stderr)
            return 2

    isolate = not args.no_isolate
    results = run_all(cases, sizes, args.repeats, args.budget, isolate)
    print(markdown(results, sizes, args.budget, args.repeats, isolate))

    if args.json:
        payload = {
            "environment": environment(),
            "sizes": sizes,
            "seconds": {k: {str(n): t for n, t in row.items()} for k, row in results.items()},
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
