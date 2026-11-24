from math import sqrt
from operator import itemgetter

import numpy as np
from numbers import Real
from scipy.stats import rankdata
from sklearn.utils import check_random_state, resample

from ..evaluation import candidate_parameters, parameter_record_conflicts
from ._score_statistics import scaled_score_moments, score_scale_limits


def subsample_cv(cv_splits, train_size, *, stratify=None, random_state=None):
    """Reduce training rows in explicit CV splits, retaining every validation row.

    Parameters
    ----------
    cv_splits : iterable of (train_indices, validation_indices)
        Materialized splits from a scikit-learn cross-validation splitter.
        Training indices must be one-dimensional integer arrays.
    train_size : float
        Fraction of each training fold to retain, in (0, 1]. A value of 1
        preserves the original splits without sampling.
    stratify : array-like of shape (n_samples,), default=None
        Full target labels for stratified training subsampling. Use this for
        classification; leave it as None for regression. Sampling that drops
        a class raises ValueError; increase the fraction in that case.
    random_state : int, RandomState instance or None, default=None
        Controls training subsampling. An integer reproduces the same splits.

    Returns
    -------
    list of tuple
        Reduced training indices paired with the original validation indices.
        Pass the original splits as ``final_selection_cv`` to re-evaluate
        finalists with full training folds before choosing the fitted winner.
        Search scores and finalist scores describe different training sizes.
    """
    if (
        isinstance(train_size, (bool, np.bool_))
        or not isinstance(train_size, Real)
        or not 0 < train_size <= 1
    ):
        raise ValueError("train_size must be a fraction in (0, 1]")
    rng = check_random_state(random_state)
    labels = None if stratify is None else np.asarray(stratify)
    if labels is not None and labels.ndim != 1:
        raise ValueError("stratify must contain one-dimensional class labels")
    result = []
    for train, validation in cv_splits:
        train = np.asarray(train)
        if train.ndim != 1 or not np.issubdtype(train.dtype, np.integer):
            raise ValueError("cv_splits must contain one-dimensional integer training indices")
        if train_size < 1:
            original_classes = None if labels is None else np.unique(labels[train])
            train = resample(
                train,
                replace=False,
                n_samples=int(len(train) * train_size),
                stratify=None if labels is None else labels[train],
                random_state=rng,
            )
            if labels is not None and not np.array_equal(
                np.unique(labels[train]), original_classes
            ):
                raise ValueError(
                    "train_size must retain every class; increase the training fraction"
                )
        result.append((train, validation))
    if not result:
        raise ValueError("cv_splits must contain at least one split")
    return result


def select_dict_keys(dictionary, keys):
    return {key: dictionary[key] for key in keys}


def _rank_scores(scores, greater_is_better=True, *, counts=None):
    """Rank candidate scores, always placing non-finite scores last.

    Candidates with finite scores are ranked first; ties share the lowest rank
    (scipy's ``method="min"``). Non-finite scores—such as NaN produced when a
    candidate fails under the default ``error_score=np.nan``—are assigned the
    worst rank so they never outrank a valid candidate.

    Parameters
    ----------
    scores : array-like
        The mean scores for each candidate. May contain non-finite values.
    greater_is_better : bool, default=True
        If True, higher scores rank better (e.g. accuracy). If False, lower
        scores rank better (e.g. an error/loss metric).
    counts : ndarray of int, optional
        Number of original candidate rows represented by each score.

    Returns
    -------
    numpy.ndarray of int
        1-based ranks, where rank 1 is the best candidate and non-finite scores rank last.
    """
    scores = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(scores)

    ranks = np.empty(scores.shape[0], dtype=int)

    if finite_mask.any():
        finite_scores = scores[finite_mask]
        ordered = -finite_scores if greater_is_better else finite_scores
        if counts is None:
            ranks[finite_mask] = rankdata(ordered, method="min").astype(int)
            ranks[~finite_mask] = finite_mask.sum() + 1
        else:
            weights = counts[finite_mask]
            order = np.argsort(ordered)
            ordered = ordered[order]
            sorted_weights = weights[order]
            sorted_ranks = np.cumsum(sorted_weights) - sorted_weights + 1
            # Equal scores from different reductions must use the same rank.
            sorted_ranks[1:][ordered[1:] == ordered[:-1]] = 0
            distinct_ranks = np.empty_like(sorted_ranks)
            distinct_ranks[order] = np.maximum.accumulate(sorted_ranks)
            ranks[finite_mask] = distinct_ranks
            ranks[~finite_mask] = weights.sum() + 1
    else:
        # All scores are non-finite: keep a deterministic, non-crashing ranking.
        ranks[:] = 1

    return ranks


def _add_cv_statistics(
    cv_results, records, return_train_score, metrics, greater_is_better, *, unique_candidates=None
):
    n_splits = len(records[0]["cv_scores"])
    failed = np.fromiter(
        (row["fit_time"] is None for row in records), dtype=bool, count=len(records)
    )
    n_failed = np.count_nonzero(failed)
    cv_results["all_fits_failed"] = failed
    # Building the identity map pays off when most rows reused a cached score.
    reuse_rows = unique_candidates is not None and 4 * unique_candidates <= len(records)
    for metric in metrics:
        for subset in ("test", "train") if return_train_score else ("test",):
            name = f"{subset}_{metric}"
            scores = [row[name] for row in records]
            positions = None
            if reuse_rows:
                scores, positions = _reused_score_rows(scores, n_splits)
            split_scores = [[] for _ in range(n_splits)]
            means, deviations = _row_statistics(
                scores, n_splits, np.mean, np.std, split_scores=split_scores
            )
            if positions is None:
                ranks = _rank_scores(means, greater_is_better)
            else:
                indices = np.asarray(positions, dtype=np.intp)
                ranks = _rank_scores(means, greater_is_better, counts=np.bincount(indices))[indices]
                expand = itemgetter(*positions)
                split_scores = [list(expand(column)) for column in split_scores]
                means, deviations = list(expand(means)), list(expand(deviations))
            for split, column in enumerate(split_scores):
                cv_results[f"split{split}_{name}"] = column
            cv_results[f"mean_{name}"] = means
            cv_results[f"std_{name}"] = deviations
            if n_failed:
                ranks[failed] = len(records) - n_failed + 1
            cv_results[f"rank_{name}"] = ranks

    for name in ("fit_time", "score_time"):
        rows = [row[name] for row in records]
        if any(row is None for row in rows):
            known = [row for row in rows if row is not None]
            means, deviations = _row_statistics(
                known, n_splits, np.nanmean, np.nanstd, reuse_rows=reuse_rows
            )
            observed_means = iter(means)
            observed_deviations = iter(deviations)
            means = [np.nan if row is None else next(observed_means) for row in rows]
            deviations = [np.nan if row is None else next(observed_deviations) for row in rows]
        else:
            means, deviations = _row_statistics(
                rows, n_splits, np.nanmean, np.nanstd, reuse_rows=reuse_rows
            )
        cv_results[f"mean_{name}"] = means
        cv_results[f"std_{name}"] = deviations


def _fold_score_moments(matrix):
    if matrix.dtype.kind == "f" and matrix.shape[1]:
        _, lower, upper = score_scale_limits(matrix.dtype)
        scales = np.max(np.abs(matrix), axis=1)
        extreme = ((scales > 0) & (scales < lower)) | (
            (scales >= upper / sqrt(matrix.shape[1])) & np.isfinite(scales)
        )
        if extreme.any():
            averages = np.empty(len(matrix), dtype=matrix.dtype)
            deviations = np.empty_like(averages)
            ordinary = matrix[~extreme]
            averages[~extreme] = np.mean(ordinary, axis=1)
            deviations[~extreme] = np.std(ordinary, axis=1)
            for index in np.flatnonzero(extreme):
                row, scale = matrix[index], scales[index]
                averages[index], deviations[index] = scaled_score_moments(row, scale)
            return averages, deviations
    return np.mean(matrix, axis=1), np.std(matrix, axis=1)


def _reused_score_rows(rows, n_splits):
    if len(rows) < 64:
        return rows, None
    # Cache hits share score arrays, even when the identical arrays appear
    # in different blocks. Equal values from separate evaluations stay separate.
    identities = list(map(id, rows))
    unique = dict(zip(identities, rows))
    if 4 * len(unique) > len(rows):
        return rows, None
    distinct = list(unique.values())
    dtype = getattr(distinct[0], "dtype", None)
    if not all(
        type(row) is np.ndarray
        and row.shape == (n_splits,)
        and row.dtype.kind == "f"
        and row.dtype == dtype
        for row in distinct
    ):
        return rows, None
    positions = dict(zip(unique, range(len(unique))))
    return distinct, tuple(positions[identity] for identity in identities)


def _row_statistics(rows, n_splits, mean, std, *, split_scores=None, reuse_rows=False):
    means, deviations = [], []
    fold_scores = mean is np.mean and std is np.std
    nan_aware = mean is np.nanmean and std is np.nanstd
    expand = None
    if reuse_rows:
        rows, positions = _reused_score_rows(rows, n_splits)
        if positions is not None:
            expand = itemgetter(*positions)
    columns = [[] for _ in range(n_splits)] if expand is not None and split_scores else split_scores
    # Bound temporary matrices for long searches and leave-one-out CV.
    batch_size = max(1, 65536 // max(1, n_splits))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        first = batch[0]
        if all(
            type(row) is np.ndarray
            and row.shape == (n_splits,)
            and row.dtype.kind == "f"
            and row.dtype == first.dtype
            for row in batch
        ):
            matrix = np.asarray(batch)
            if columns is not None:
                for column, values in zip(columns, matrix.T):
                    column.extend(values)
            if fold_scores:
                averages, spreads = _fold_score_moments(matrix)
            elif nan_aware and matrix.dtype == np.float64 and np.isfinite(matrix).all():
                # Ordinary CV timings need no missing-value buffers or counts.
                averages, spreads = np.mean(matrix, axis=1), np.std(matrix, axis=1)
            else:
                averages, spreads = mean(matrix, axis=1), std(matrix, axis=1)
            means.extend(averages)
            deviations.extend(spreads)
        else:
            # Preserve individual dtypes and reductions for legacy logbooks.
            if columns is not None:
                for split, column in enumerate(columns):
                    column.extend([row[split] for row in batch])
            if fold_scores:
                for row in batch:
                    averages, spreads = _fold_score_moments(np.asarray(row).reshape(1, -1))
                    means.append(averages[0])
                    deviations.append(spreads[0])
            else:
                means.extend(mean(row) for row in batch)
                deviations.extend(std(row) for row in batch)
    if expand is not None:
        if split_scores:
            for output, column in zip(split_scores, columns, strict=True):
                output.extend(expand(column))
        return list(expand(means)), list(expand(deviations))
    return means, deviations


def create_gasearch_cv_results_(
    records, space, return_train_score, metrics, greater_is_better=True, *, unique_candidates=None
):
    names = space.parameters
    parameter_rows = records
    if parameter_record_conflicts(names, metrics, return_train_score):
        parameter_rows = list(map(candidate_parameters, records))
    cv_results = {f"param_{name}": [row[name] for row in parameter_rows] for name in names}
    _add_cv_statistics(
        cv_results,
        records,
        return_train_score,
        metrics,
        greater_is_better,
        unique_candidates=unique_candidates,
    )
    if len(names) == 1:
        (name,) = names
        cv_results["params"] = [{name: value} for value in cv_results[f"param_{name}"]]
    else:
        cv_results["params"] = [select_dict_keys(row, names) for row in parameter_rows]
    return cv_results


def _shared_feature_counts(masks):
    # Cached rows can share the same Boolean array. Legacy masks retain their
    # own sum implementation and are counted separately.
    counts_by_identity = {}
    counts = []
    for mask in masks:
        identity = id(mask)
        count = counts_by_identity.get(identity)
        if count is None:
            if type(mask) is np.ndarray and mask.dtype.kind == "b":
                count = np.int_(np.count_nonzero(mask))
                counts_by_identity[identity] = count
            else:
                count = np.sum(mask)
        counts.append(count)
    return counts


def create_feature_selection_cv_results_(
    records, return_train_score, metrics, greater_is_better=True, *, unique_candidates=None
):
    cv_results = {}
    _add_cv_statistics(
        cv_results,
        records,
        return_train_score,
        metrics,
        greater_is_better,
        unique_candidates=unique_candidates,
    )
    features = [row["features"] for row in records]
    if unique_candidates is not None and len(features) >= max(64, 4 * unique_candidates):
        cv_results["n_features"] = _shared_feature_counts(features)
    else:
        cv_results["n_features"] = [
            (
                np.int_(np.count_nonzero(mask))
                if type(mask) is np.ndarray and mask.dtype.kind == "b"
                else np.sum(mask)
            )
            for mask in features
        ]
    cv_results["rank_n_features"] = rankdata(cv_results["n_features"], method="min").astype(int)
    cv_results["features"] = features
    return cv_results
