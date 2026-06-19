from joblib import Parallel, delayed, effective_n_jobs


def logbook_record(logbook, chapter_name, parameters):
    index = len(logbook.chapters[chapter_name])
    parameters = {"index": index, **parameters}
    logbook.record(**{chapter_name: parameters})
    return parameters


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
    }


def validate_parallel_backend(parallel_backend):
    valid_backends = {"auto", "population", "cv"}
    if parallel_backend not in valid_backends:
        raise ValueError(
            f"parallel_backend must be one of {sorted(valid_backends)}, "
            f"got {parallel_backend} instead"
        )


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


def evaluate_population(estimator, individuals, cache_record_key):
    if not individuals:
        return []

    pending_items = []
    pending_lookup_keys = []
    seen_pending_keys = set()
    batch_cache_hits = 0
    batch_duplicates = 0

    for individual in individuals:
        individual_key = estimator._individual_key(individual)
        if estimator.use_cache and individual_key in estimator.fitness_cache:
            batch_cache_hits += 1
            pending_lookup_keys.append(None)
            continue

        if estimator.use_cache and individual_key in seen_pending_keys:
            batch_duplicates += 1
            pending_lookup_keys.append(individual_key)
            continue

        lookup_key = individual_key if estimator.use_cache else (individual_key, len(pending_items))
        pending_items.append((lookup_key, list(individual)))
        pending_lookup_keys.append(lookup_key)
        seen_pending_keys.add(individual_key)

    pending_results = {}

    if pending_items:
        if use_population_parallelism(estimator, len(pending_items)):
            estimator.fit_stats_["population_parallel_batches"] += 1
            results = Parallel(n_jobs=estimator.n_jobs, prefer="threads")(
                delayed(estimator._evaluate_individual)(individual, n_jobs=1)
                for _, individual in pending_items
            )
        else:
            estimator.fit_stats_["population_serial_batches"] += 1
            candidate_n_jobs = estimator.n_jobs if estimator.parallel_backend == "cv" else 1
            results = [
                estimator._evaluate_individual(individual, n_jobs=candidate_n_jobs)
                for _, individual in pending_items
            ]

        pending_results = {
            lookup_key: result for (lookup_key, _), result in zip(pending_items, results)
        }

    fitnesses = []
    batch_cv_calls = 0
    batch_skipped = 0

    for individual, lookup_key in zip(individuals, pending_lookup_keys):
        individual_key = estimator._individual_key(individual)

        if estimator.use_cache and individual_key in estimator.fitness_cache:
            cached_result = estimator.fitness_cache[individual_key]
            estimator.logbook.record(parameters=cached_result[cache_record_key])
        else:
            fitness, current_generation_record, used_cv, skipped_invalid = pending_results[
                lookup_key
            ]
            current_generation_record = logbook_record(
                estimator.logbook,
                "parameters",
                current_generation_record,
            )
            batch_cv_calls += int(used_cv)
            batch_skipped += int(skipped_invalid)
            if estimator.use_cache:
                estimator.fitness_cache[individual_key] = {
                    "fitness": fitness,
                    cache_record_key: current_generation_record,
                }

        fitnesses.append(
            estimator.fitness_cache[individual_key]["fitness"] if estimator.use_cache else fitness
        )

    record_fit_stats(
        estimator,
        evaluated=len(individuals),
        unique=len(pending_items),
        cv_calls=batch_cv_calls,
        cache_hits=batch_cache_hits,
        duplicates=batch_duplicates,
        skipped=batch_skipped,
    )

    return fitnesses
