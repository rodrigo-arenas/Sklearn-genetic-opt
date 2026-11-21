"""Generation-boundary persistence for the bundled optimizer adapters."""

from functools import partial, wraps
import pickle
import platform
import random
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from joblib import hash as joblib_hash

from ._version import __version__


def checkpoint_engine(optimizer):
    if isinstance(optimizer, partial):
        optimizer = optimizer.func
    return getattr(optimizer, "_checkpoint_engine", None)


def resume_method(optimizer):
    method = getattr(optimizer, "resume", None)
    if method is None and isinstance(optimizer, partial) and checkpoint_engine(optimizer):
        # Forward the same documented solver options as the original fit.
        return partial(optimizer.func.resume, **optimizer.keywords)
    return method


def environment(engine):
    try:
        engine_version = version(engine)
    except PackageNotFoundError as exc:
        raise ImportError(f"Install sklearn-genetic-opt[{engine}] to use this optimizer") from exc
    packages = {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}
    packages[engine] = engine_version
    return (1, __version__, sys.platform, platform.machine(), platform.python_version(), packages)


def validate_boundary(engine, checkpoint):
    """Cross-check the solver cursor with the estimator before replaying any work."""
    log = checkpoint["logbook"]
    state = checkpoint["solver_state"]
    if engine in ("cmaes", "nevergrad"):
        valid = isinstance(state, tuple) and len(state) == 2 and state[0] == len(log) - 1
    else:
        valid = (
            isinstance(state, list)
            and len(state) == len(log)
            and all(
                len(batch) == 3
                and len(batch[1]) == record["nevals"]
                and len(batch[2]) == record["population_size"]
                for batch, record in zip(state, log)
            )
        )
    if not valid or sum(record["nevals"] for record in log) != len(checkpoint["candidate_records"]):
        raise ValueError(
            "Optimizer checkpoint state does not match the recorded generation boundary"
        )


def native_checkpoint(engine):
    """Restore the private solver only after the runtime validates its environment."""

    def decorate(optimize):
        def resume(problem, evaluate, report, state, **options):
            generation, payload = state
            # A cold optional import (notably Nevergrad) can consume NumPy's
            # global RNG. Preserve the saved streams for CV, observers and asks.
            rng = random.getstate(), np.random.get_state()
            try:
                import_module(engine)
                restored = pickle.loads(payload)
            finally:
                random.setstate(rng[0])
                np.random.set_state(rng[1])
            optimize(problem, evaluate, report, _state=(generation, *restored), **options)

        optimize.resume = resume
        optimize._checkpoint_engine = engine
        return optimize

    return decorate


def report_native(problem, report, handles, state):
    if problem.checkpoint_requested:
        generation, *payload = state
        report(handles, state=(generation, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)))
    else:
        report(handles)


def replay_checkpoint(engine):
    """Replay the deterministic, independently seeded batches of SciPy and pymoo."""

    def decorate(solve):
        @wraps(solve)
        def optimize(problem, evaluate, report):
            if getattr(problem, "checkpoint_requested", False):
                return resume(problem, evaluate, report, [])
            return solve(problem, evaluate, report)

        def resume(problem, evaluate, report, journal):
            # Both adapters evaluate one batch before each survivor report. Opaque
            # handles share the snapshot pickle with the archive and runtime owner.
            boundary = len(journal)
            saved_rng = (random.getstate(), np.random.get_state()) if boundary else None
            generation = 0
            pending = None

            def batch(rows):
                nonlocal pending
                if pending is not None:
                    raise ValueError(
                        "Expected one evaluation batch per optimizer checkpoint report"
                    )
                rows = tuple(map(tuple, rows))
                signature = joblib_hash(rows)
                if generation < boundary:
                    expected, replies, _ = journal[generation]
                    if expected != signature:
                        raise ValueError(
                            "Optimizer checkpoint replay diverged on candidate proposals"
                        )
                else:
                    replies = evaluate(rows)
                pending = (signature, replies)
                return replies

            def finish(survivors):
                nonlocal generation, pending, saved_rng
                survivors = tuple(survivors)
                if pending is None:
                    raise ValueError("Missing evaluation batch during optimizer checkpoint replay")
                if generation < boundary:
                    if survivors != journal[generation][2]:
                        raise ValueError("Optimizer checkpoint replay diverged on survivors")
                    if generation + 1 == boundary:
                        # The engines have independent proposal RNGs. Restore the
                        # caller's RNGs before any fresh evaluation or callback.
                        random.setstate(saved_rng[0])
                        np.random.set_state(saved_rng[1])
                        saved_rng = None
                else:
                    journal.append((*pending, survivors))
                    report(survivors, state=journal)
                generation += 1
                pending = None

            try:
                solve(problem, batch, finish)
            finally:
                if saved_rng is not None:
                    random.setstate(saved_rng[0])
                    np.random.set_state(saved_rng[1])
            if generation < boundary:
                raise ValueError("Optimizer stopped before the saved checkpoint boundary")

        optimize.resume = resume
        optimize._checkpoint_engine = engine
        return optimize

    return decorate
