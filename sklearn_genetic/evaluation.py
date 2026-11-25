from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import partial
from itertools import chain, islice
from numbers import Real

from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
from joblib.parallel import ThreadingBackend, get_active_backend
from sklearn import config_context, get_config

_CANDIDATE_PARAMS_FIELD = "_candidate_params"
_MISSING = object()


class _CandidateParameters(dict):
    """Distinguish stored parameters from a historical parameter of the same name."""


def parameter_record_conflicts(names, metrics, return_train_score):
    fields = {"index", "score", "cv_scores", "fit_time", "score_time"}
    fields.update(f"test_{metric}" for metric in metrics)
    if return_train_score:
        fields.update(f"train_{metric}" for metric in metrics)
    return fields.intersection(names)


def preserve_candidate_parameters(params, conflicts):
    row = {name: value for name, value in params.items() if name not in conflicts}
    row[_CANDIDATE_PARAMS_FIELD] = _CandidateParameters(params)
    return row


def candidate_parameters(row):
    params = row.get(_CANDIDATE_PARAMS_FIELD)
    return params if type(params) is _CandidateParameters else row


def validate_error_score(error_score):
    if isinstance(error_score, str):
        if error_score == "raise":
            return
        raise ValueError(f"error_score must be numeric or 'raise', got {error_score!r} instead")

    if isinstance(error_score, Real):
        return

    raise ValueError(f"error_score must be numeric or 'raise', got {error_score!r} instead")


def logbook_record(logbook, chapter_name, parameters):
    index = len(logbook.chapters[chapter_name])
    # Keep the index first, but replace any index from a cached source row.
    parameters = {"index": index, **parameters}
    parameters["index"] = index
    # Dictionary-valued parameters are values, not nested DEAP chapters.
    logbook.chapters[chapter_name].append(parameters)
    logbook.record()
    return parameters


def candidate_records(estimator):
    if "_candidate_rows" in vars(estimator):
        return estimator._candidate_rows
    return estimator.logbook.chapters["parameters"]


def record_candidate(estimator, parameters):
    if "_candidate_rows" in vars(estimator) and "logbook" not in vars(estimator):
        rows = estimator._candidate_rows
        index = len(rows)
        record = {"index": index, **parameters}
        record["index"] = index
        rows.append(record)
        return record
    return logbook_record(estimator.logbook, "parameters", parameters)


def restore_candidate_positions(estimator, records):
    if "index" in getattr(estimator, "_candidate_parameter_conflicts", ()):
        # An older record may store a literal parameter named "index" there.
        # Move it into the parameter map before replacing the diagnostic field.
        cached_records = (
            result["current_generation_params"] for result in estimator.fitness_cache.values()
        )
        for row in chain(records, cached_records):
            if candidate_parameters(row) is row:
                row[_CANDIDATE_PARAMS_FIELD] = _CandidateParameters(
                    (name, row[name]) for name in estimator.space.parameters
                )
    for index, row in enumerate(records):
        row["index"] = index


def is_parallel_enabled(n_jobs, n_tasks):
    return n_tasks > 1 and effective_n_jobs(n_jobs) != 1


def create_fit_stats():
    return {
        "evaluated_candidates": 0,
        "unique_candidates": 0,
        "cross_validate_calls": 0,
        "cache_hits": 0,
        "duplicate_candidates": 0,
        "skipped_invalid_candidates": 0,
        "population_parallel_batches": 0,
        "population_serial_batches": 0,
        "random_immigrants": 0,
        "local_refinement_candidates": 0,
    }


def validate_parallel_backend(parallel_backend, *, n_jobs=None):
    valid_backends = {"auto", "population", "cv"}
    if parallel_backend not in valid_backends:
        raise ValueError(
            f"parallel_backend must be one of {sorted(valid_backends)}, "
            f"got {parallel_backend} instead"
        )
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero. Use n_jobs=1 for serial execution.")


def use_population_parallelism(estimator, n_tasks):
    if estimator.parallel_backend == "cv":
        return False

    return (
        estimator.log_config is None
        and is_parallel_enabled(estimator.n_jobs, n_tasks)
        and estimator.parallel_backend in {"auto", "population"}
    )


def record_fit_stats(
    estimator, evaluated=0, unique=0, cv_calls=0, cache_hits=0, duplicates=0, skipped=0
):
    estimator.fit_stats_["evaluated_candidates"] += evaluated
    estimator.fit_stats_["unique_candidates"] += unique
    estimator.fit_stats_["cross_validate_calls"] += cv_calls
    estimator.fit_stats_["cache_hits"] += cache_hits
    estimator.fit_stats_["duplicate_candidates"] += duplicates
    estimator.fit_stats_["skipped_invalid_candidates"] += skipped


def _configured_cv_limit(estimator):
    from .callbacks.early_stoppers import CVBudgetStopping, _validate_positive_integer

    limits = []
    for callback in estimator.callbacks:
        if isinstance(callback, CVBudgetStopping):
            limit = callback.max_cv_calls
            _validate_positive_integer(limit, "max_cv_calls")
            limits.append(int(limit))
    return estimator.fit_stats_["cross_validate_calls"] + min(limits) if limits else None


def start_evaluation_callbacks(callbacks, estimator):
    from .callbacks.validations import eval_callbacks

    eval_callbacks(callbacks, None, None, estimator, "on_start")
    if getattr(estimator, "_evaluation_cv_limit", None) is not None:
        # A budget callback can choose its limit from the materialized folds.
        estimator._evaluation_cv_limit = _configured_cv_limit(estimator)


@contextmanager
def evaluation_context(estimator):
    from .callbacks.early_stoppers import TimerStopping

    estimator._evaluation_cv_limit = _configured_cv_limit(estimator)
    estimator._evaluation_timers = [c for c in estimator.callbacks if isinstance(c, TimerStopping)]
    estimator._evaluation_started = False
    estimator._evaluation_budget_exhausted = False
    estimator._population_pool = None
    try:
        yield
    finally:
        pool = estimator._population_pool
        del (
            estimator._evaluation_timers,
            estimator._evaluation_cv_limit,
            estimator._evaluation_started,
            estimator._evaluation_budget_exhausted,
            estimator._population_pool,
        )
        if pool is not None:
            pool[1].shutdown(cancel_futures=True)


def _cv_calls_remaining(estimator):
    limit = getattr(estimator, "_evaluation_cv_limit", None)
    return None if limit is None else max(0, limit - estimator.fit_stats_["cross_validate_calls"])


def _evaluation_limit_reached(estimator):
    return _cv_calls_remaining(estimator) == 0 or _time_budget_exhausted(estimator)


def _time_budget_exhausted(estimator):
    if getattr(estimator, "_evaluation_budget_exhausted", False):
        return True
    # Complete at least one initial batch, even if setup already used the budget.
    if not getattr(estimator, "_evaluation_started", False):
        return False
    if any(timer._time_exceeded() for timer in getattr(estimator, "_evaluation_timers", ())):
        estimator._evaluation_budget_exhausted = True
        return True
    return False


def _evaluate_in_thread(evaluate, backend, individual):
    nested, n_jobs = backend.get_nested_backend()
    with parallel_backend(backend=nested, n_jobs=n_jobs):
        return evaluate(individual)


def _evaluate_with_config(evaluate, config, individual):
    with config_context(**config):
        return evaluate(individual, n_jobs=1)


def _ordered_thread_results(pool, workers, evaluate, individuals, *, pending):
    # Keep a bounded number of tasks in flight even for very large populations.
    remaining = iter(individuals)
    pending.extend(pool.submit(evaluate, row) for row in islice(remaining, 2 * workers))
    for row in remaining:
        result = pending[0].result()
        pending.popleft()
        yield result
        pending.append(pool.submit(evaluate, row))
    while pending:
        result = pending[0].result()
        pending.popleft()
        yield result


def _count_completed_cv_calls(results):
    calls = 0
    for result in results:
        try:
            calls += int(result[2])
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            # A malformed evaluator result cannot confirm work during failure handling.
            continue
    return calls


@contextmanager
def _population_dispatch(estimator, evaluate):
    parallel = Parallel(n_jobs=estimator.n_jobs, prefer="threads")
    # scikit-learn config is thread-local and joblib workers can be reused.
    evaluate = partial(_evaluate_with_config, evaluate, get_config())
    if hasattr(estimator, "_population_pool"):
        backend, _ = get_active_backend(prefer="threads")
        # Preserve explicitly selected process/custom backends, nested scheduling,
        # and joblib progress reporting. The fast path is the ordinary thread backend.
        if (
            type(backend) is ThreadingBackend
            and backend.nesting_level == 0
            and not parallel.verbose
        ):
            workers = backend.effective_n_jobs(parallel.n_jobs)
            if workers > 1:
                pool = estimator._population_pool
                if pool is None or pool[0] != workers:
                    if pool is not None:
                        pool[1].shutdown()
                    pool = workers, ThreadPoolExecutor(max_workers=workers)
                    estimator._population_pool = pool
                threaded_evaluate = partial(_evaluate_in_thread, evaluate, backend)
                pending = deque()
                try:
                    yield partial(
                        _ordered_thread_results,
                        pool[1],
                        workers,
                        threaded_evaluate,
                        pending=pending,
                    )
                except BaseException:
                    pool[1].shutdown(cancel_futures=True)
                    estimator._population_pool = None
                    if "_unrecorded_search_cv_calls" in vars(estimator):
                        # Workers can finish later proposals while ordered delivery
                        # waits for the proposal that ultimately raises.
                        completed = (
                            future.result()
                            for future in pending
                            if not future.cancelled() and future.exception() is None
                        )
                        estimator._unrecorded_search_cv_calls += _count_completed_cv_calls(
                            completed
                        )
                    raise
                return
    with parallel:
        yield lambda individuals: parallel(delayed(evaluate)(row) for row in individuals)


def _pending_cache_positions(estimator, individuals, lookup_keys, repeated_results, mask_repeats):
    """Store ordered lookup keys, mark repeats and yield first uncached positions."""
    first_positions = {}
    for position, individual in enumerate(individuals):
        lookup_key = estimator._individual_key(individual)
        first = mask_repeats.get(position) if mask_repeats else None
        # The actual lookup keys must also share identity after canonicalization.
        if first is None or lookup_keys[first] is not lookup_key:
            first = first_positions.setdefault(lookup_key, position)
        if first == position:
            lookup_keys.append(lookup_key)
            if lookup_key not in estimator.fitness_cache:
                yield position
        else:
            shared_key = lookup_keys[first]
            lookup_keys.append(shared_key)
            repeated_results[id(shared_key)] = _MISSING


def evaluate_population(
    estimator, individuals, cache_record_key, *, evaluate=None, mask_repeats=None
):
    if not individuals:
        return []

    use_cache = estimator.use_cache
    remaining = _cv_calls_remaining(estimator)
    timers = getattr(estimator, "_evaluation_timers", ())
    pending_scan = None
    repeated_results = {}
    if use_cache:
        lookup_keys = []
        positions = _pending_cache_positions(
            estimator, individuals, lookup_keys, repeated_results, mask_repeats
        )
        if timers:
            # Two fresh candidates suffice to decide whether to use population
            # workers. A fully spent budget needs only the next fresh candidate.
            pending_scan = positions
            positions = islice(positions, 1 if remaining == 0 else 2)
        elif remaining is not None and remaining < len(individuals):
            # One extra fresh proposal marks the end of the recordable prefix,
            # including cached rows that follow the last permitted CV call.
            pending_scan = positions
            positions = islice(positions, remaining + 1)
        pending_positions = list(positions)
    else:
        pending_positions = range(len(individuals))

    results = []
    cv_calls_before = estimator.fit_stats_["cross_validate_calls"]
    try:
        dispatchable = len(pending_positions)
        if remaining is not None:
            dispatchable = min(dispatchable, remaining)
        if dispatchable:
            if evaluate is None:
                evaluate = estimator._evaluate_individual
            reuse_mask_keys = False
            if (
                use_cache
                and cache_record_key == "current_generation_features"
                and len(individuals[0]) >= 4096
            ):
                from .genetic_search import _uses_cached_feature_mask

                reuse_mask_keys = _uses_cached_feature_mask(estimator, evaluate)
                if reuse_mask_keys:
                    evaluate = partial(evaluate, from_cache_key=True)
            population_parallel = use_population_parallelism(estimator, dispatchable)
            candidate_n_jobs = estimator.n_jobs if estimator.parallel_backend == "cv" else 1
            batch_size = dispatchable
            if timers:
                batch_size = effective_n_jobs(estimator.n_jobs) if population_parallel else 1
            executor = (
                _population_dispatch(estimator, evaluate) if population_parallel else nullcontext()
            )
            start = 0
            with executor as parallel:
                while start < len(pending_positions) and remaining != 0:
                    size = batch_size if remaining is None else min(batch_size, remaining)
                    if pending_scan is not None and timers:
                        # One fresh lookahead distinguishes cached rows at the end
                        # of this dispatch from rows after the first unfinished one.
                        to_scan = start + size + 1 - len(pending_positions)
                        if to_scan > 0:
                            pending_positions.extend(islice(pending_scan, to_scan))
                    if _evaluation_limit_reached(estimator):
                        break
                    batch_positions = pending_positions[start : start + size]
                    if reuse_mask_keys:
                        # The native evaluator builds a private Boolean mask from the immutable key.
                        batch = (lookup_keys[i] for i in batch_positions)
                    else:
                        # Copy one row at a time as the executor consumes the batch.
                        batch = (list(individuals[i]) for i in batch_positions)
                    if population_parallel:
                        batch_results = parallel(batch)
                    else:
                        batch_results = (
                            evaluate(individual, n_jobs=candidate_n_jobs) for individual in batch
                        )
                    if remaining is None:
                        results.extend(batch_results)
                    else:
                        # Reserve one potential CV call per candidate before dispatch.
                        # An invalid candidate that skips CV leaves its slot available.
                        for result in batch_results:
                            results.append(result)
                            remaining -= int(result[2])
                    start += len(batch_positions)
                    if hasattr(estimator, "_evaluation_started"):
                        estimator._evaluation_started = True
                    if pending_scan is not None and not timers and remaining:
                        # A skipped CV frees a slot. Discover only enough proposals
                        # to fill the remaining slots and keep one lookahead.
                        to_scan = start + remaining + 1 - len(pending_positions)
                        if to_scan > 0:
                            pending_positions.extend(islice(pending_scan, to_scan))
            if results:
                stat = (
                    "population_parallel_batches"
                    if population_parallel
                    else "population_serial_batches"
                )
                estimator.fit_stats_[stat] += 1

        if use_cache:
            # Repeated proposals share a key object within this batch.
            # Keep full value keys in the persistent cache and use identities here.
            pending_results = {
                id(lookup_keys[position]): result
                for position, result in zip(pending_positions, results)
            }
        else:
            pending_results = results
            lookup_keys = range(len(results))

        fitnesses = []
        batch_cv_calls = 0
        batch_cache_hits = 0
        batch_skipped = 0

        for lookup_key in lookup_keys:
            pending_key = id(lookup_key) if use_cache else lookup_key
            cached_result = _MISSING
            # Repeated keys start marked as missing until their first row is recorded.
            repeated_result = None
            if use_cache:
                repeated_result = repeated_results.get(pending_key)
                if repeated_result is None or repeated_result is _MISSING:
                    cached_result = estimator.fitness_cache.get(lookup_key, _MISSING)
                else:
                    cached_result = repeated_result
            if cached_result is not _MISSING:
                record_candidate(estimator, cached_result[cache_record_key])
                fitness = cached_result["fitness"]
                # The first occurrence may have populated the cache in this batch.
                batch_cache_hits += pending_key not in pending_results
            else:
                if use_cache:
                    result = pending_results.get(pending_key, _MISSING)
                    if result is _MISSING:
                        break
                else:
                    result = pending_results[pending_key]
                fitness, current_generation_record, used_cv, skipped_invalid = result
                current_generation_record = record_candidate(
                    estimator,
                    current_generation_record,
                )
                batch_cv_calls += int(used_cv)
                batch_skipped += int(skipped_invalid)
                if use_cache:
                    cached_result = {
                        "fitness": fitness,
                        cache_record_key: current_generation_record,
                    }
                    estimator.fitness_cache[lookup_key] = cached_result

            if repeated_result is _MISSING:
                repeated_results[pending_key] = cached_result

            fitnesses.append(fitness)

        # Keep only completed individuals; callers must not assign phantom fitnesses
        # or retain unevaluated candidates when a limit truncates the population.
        if len(fitnesses) < len(individuals):
            del individuals[len(fitnesses) :]
        if fitnesses and hasattr(estimator, "_evaluation_started"):
            estimator._evaluation_started = True
        batch_duplicates = len(fitnesses) - len(pending_results) - batch_cache_hits
        record_fit_stats(
            estimator,
            evaluated=len(fitnesses),
            unique=len(pending_results),
            cv_calls=batch_cv_calls,
            cache_hits=batch_cache_hits,
            duplicates=batch_duplicates,
            skipped=batch_skipped,
        )

        return fitnesses
    except BaseException:
        if results and "_unrecorded_search_cv_calls" in vars(estimator):
            # A batch can fail after receiving its results or during publication.
            # Count only the completed calls the optimizer's counters still lack.
            recorded = estimator.fit_stats_["cross_validate_calls"] - cv_calls_before
            completed = _count_completed_cv_calls(results)
            estimator._unrecorded_search_cv_calls += max(0, completed - recorded)
        raise
