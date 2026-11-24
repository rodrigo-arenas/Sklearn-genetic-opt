"""Generation-boundary persistence for the bundled optimizer adapters."""

from collections.abc import Mapping
from functools import partial, wraps
from hashlib import md5
from itertools import chain, repeat
from numbers import Integral
from operator import is_
import pickle
import platform
import random
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from joblib import hash as joblib_hash

from ._version import __version__


def training_data_hash(estimator):
    X = estimator.X_
    if getattr(estimator, "preserve_dataframe", False):
        # Normalize the pandas block layout so an equivalent copy can resume.
        # Keep the index, columns, data and dtypes in the fingerprint.
        X = X.copy(deep=True)
    return joblib_hash((type(estimator), X, estimator.y_, estimator.groups_, estimator._fit_params))


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


def _is_nonnegative_count(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0


def _valid_generation_record(record, generation):
    return (
        isinstance(record, Mapping)
        and _is_nonnegative_count(record.get("gen"))
        and record["gen"] == generation
        and _is_nonnegative_count(record.get("nevals"))
        and _is_nonnegative_count(record.get("population_size"))
        and record["population_size"] > 0
    )


def validate_boundary(engine, checkpoint):
    """Cross-check the solver cursor with the estimator before replaying any work."""
    log = checkpoint["logbook"]
    if (
        not isinstance(log, (list, tuple))
        or not log
        or not all(
            _valid_generation_record(record, generation) for generation, record in enumerate(log)
        )
    ):
        raise ValueError("Optimizer checkpoint logbook has an invalid generation boundary")
    records = checkpoint["candidate_records"]
    if not isinstance(records, list) or sum(int(row["nevals"]) for row in log) != len(records):
        raise ValueError(
            "Optimizer checkpoint candidate_records do not match the recorded generation boundary"
        )
    state = checkpoint["solver_state"]
    if engine in ("cmaes", "nevergrad"):
        valid = (
            isinstance(state, tuple)
            and len(state) == 2
            and _is_nonnegative_count(state[0])
            and state[0] == len(log) - 1
            and isinstance(state[1], bytes)
        )
    else:
        valid = (
            isinstance(state, list)
            and len(state) == len(log)
            and all(
                isinstance(batch, (list, tuple))
                and len(batch) == 3
                and isinstance(batch[0], str)
                and isinstance(batch[1], (list, tuple))
                and len(batch[1]) == record["nevals"]
                and isinstance(batch[2], tuple)
                and len(batch[2]) == record["population_size"]
                for batch, record in zip(state, log)
            )
        )
    if not valid:
        raise ValueError(
            "Optimizer checkpoint solver_state does not match the recorded generation boundary"
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
                try:
                    restored = pickle.loads(payload)
                except (pickle.UnpicklingError, EOFError, ValueError) as exc:
                    raise ValueError(
                        "Optimizer checkpoint solver_state contains unreadable data"
                    ) from exc
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


def _feature_proposal_signature(rows, *, features):
    if features and all(map(is_, map(type, chain.from_iterable(rows)), repeat(int))):
        # Use the same protocol and digest as joblib for native integer tuples.
        serialized = pickle.dumps(rows, protocol=3)
        return md5(serialized, usedforsecurity=False).hexdigest()
    return None


def replay_checkpoint(engine):
    """Replay the deterministic, independently seeded batches of SciPy and pymoo."""

    def decorate(solve):
        @wraps(solve)
        def optimize(problem, evaluate, report, **options):
            if getattr(problem, "checkpoint_requested", False):
                return resume(problem, evaluate, report, [], **options)
            return solve(problem, evaluate, report, **options)

        def resume(problem, evaluate, report, journal, **options):
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
                if type(rows) is not tuple or any(type(row) is not tuple for row in rows):
                    rows = tuple(map(tuple, rows))
                signature = _feature_proposal_signature(rows, features=problem.task == "features")
                if generation < boundary:
                    expected, replies, _ = journal[generation]
                    if (signature is None or signature != expected) and (
                        joblib_hash(rows) != expected
                    ):
                        raise ValueError(
                            "Optimizer checkpoint replay diverged on candidate proposals"
                        )
                else:
                    signature = signature or joblib_hash(rows)
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
                # A changed stopping rule must let the solver verify every saved batch.
                return generation < boundary

            try:
                solve(problem, batch, finish, **options)
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
