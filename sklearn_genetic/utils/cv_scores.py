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
        preserves the original splits without sampling. Every fold must
        retain at least one training row after rounding down.
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
    for fold, (train, validation) in enumerate(cv_splits, start=1):
        train = np.asarray(train)
        if train.ndim != 1 or not np.issubdtype(train.dtype, np.integer):
            raise ValueError("cv_splits must contain one-dimensional integer training indices")
        if not train.size:
            raise ValueError(f"cv_splits fold {fold} has no training rows")
        if train_size < 1:
            n_samples = int(train.size * train_size)
            if not n_samples:
                raise ValueError(
                    f"train_size leaves no training rows in fold {fold}. "
                    "Increase the training fraction."
                )
            original_classes = None if labels is None else np.unique(labels[train])
            train = resample(
                train,
                replace=False,
                n_samples=n_samples,
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
    scores = np.asarray(scores)
    if scores.dtype.kind != "f":
        scores = scores.astype(float)
    finite_mask = np.isfinite(scores)
    n_finite = np.count_nonzero(finite_mask)
    n_scores = scores.shape[0]
    if not n_finite:
        # All scores are non-finite. Keep a deterministic ranking.
        return np.ones(n_scores, dtype=int)

    all_finite = scores.ndim == 1 and n_finite == n_scores
    finite_scores = scores if all_finite else scores[finite_mask]
    if (
        n_finite >= 64
        and finite_scores[0] == finite_scores[-1]
        and np.all(finite_scores == finite_scores[0])
    ):
        # A constant finite score only needs the weights for the missing ranks.
        ranks = np.ones(n_scores, dtype=int)
        if not all_finite:
            finite_count = n_finite if counts is None else counts[finite_mask].sum()
            ranks[~finite_mask] = finite_count + 1
        return ranks

    ordered = -finite_scores if greater_is_better else finite_scores
    if counts is None:
        finite_ranks = rankdata(ordered, method="min")
    else:
        weights = counts if all_finite else counts[finite_mask]
        order = np.argsort(ordered)
        ordered = ordered[order]
        sorted_weights = weights[order]
        sorted_ranks = np.cumsum(sorted_weights) - sorted_weights + 1
        # Equal scores from different reductions must use the same rank.
        sorted_ranks[1:][ordered[1:] == ordered[:-1]] = 0
        finite_ranks = np.empty_like(sorted_ranks)
        finite_ranks[order] = np.maximum.accumulate(sorted_ranks)

    if all_finite:
        return finite_ranks.astype(int, copy=False)

    ranks = np.empty(n_scores, dtype=int)
    ranks[finite_mask] = finite_ranks
    finite_count = n_finite if counts is None else weights.sum()
    ranks[~finite_mask] = finite_count + 1
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


def _finite_float64_moments(matrix):
    # Reuse the row means in the centered, two-pass population variance.
    if not matrix.shape[1]:
        return np.mean(matrix, axis=1), np.std(matrix, axis=1)
    averages = matrix.mean(axis=1)
    centered = matrix - averages[:, None]
    np.multiply(centered, centered, out=centered)
    deviations = centered.mean(axis=1)
    np.sqrt(deviations, out=deviations)
    return averages, deviations


def _fold_score_moments(matrix):
    if matrix.dtype.kind == "f" and matrix.shape[1]:
        _, lower, upper = score_scale_limits(matrix.dtype)
        scales = np.max(np.abs(matrix), axis=1)
        finite = np.isfinite(scales)
        extreme = ((scales > 0) & (scales < lower)) | (
            (scales >= upper / sqrt(matrix.shape[1])) & finite
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
        if matrix.dtype == np.float64 and finite.all():
            return _finite_float64_moments(matrix)
    return np.mean(matrix, axis=1), np.std(matrix, axis=1)


def _reused_score_rows(rows, n_splits):
    if len(rows) < 64:
        return rows, None
    # Cache hits share score arrays, even when the identical arrays appear
    # in different blocks. Equal values from separate evaluations stay separate.
    limit = len(rows) // 4
    by_identity = {}
    distinct, positions = [], []
    next_position = 0
    for row in rows:
        position = by_identity.setdefault(id(row), next_position)
        if position == next_position:
            # This identity is new. Once there are too many, counting the
            # remaining rows cannot make the reuse path eligible.
            if next_position == limit:
                return rows, None
            distinct.append(row)
            next_position += 1
        positions.append(position)
    dtype = getattr(distinct[0], "dtype", None)
    if not all(
        type(row) is np.ndarray
        and row.shape == (n_splits,)
        and row.dtype.kind == "f"
        and row.dtype == dtype
        for row in distinct
    ):
        return rows, None
    return distinct, tuple(positions)


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
                averages, spreads = _finite_float64_moments(matrix)
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


def _rank_feature_counts(counts):
    if len(counts) >= 128:
        counts = np.asarray(counts)
        if (
            counts.ndim == 1
            and counts.dtype.kind == "i"
            and counts.min() >= 0
            and counts.max() <= len(counts)
        ):
            # The number of smaller masks determines each minimum tied rank.
            # Bound the histogram by the number of candidate rows.
            frequency = np.bincount(counts)
            ranks_by_size = np.cumsum(frequency) - frequency + 1
            return ranks_by_size[counts].astype(int, copy=False)
    return rankdata(counts, method="min").astype(int)


def create_feature_selection_cv_results_(
    records,
    return_train_score,
    metrics,
    greater_is_better=True,
    *,
    unique_candidates=None,
    mask_limit=None,
):
    features = [row["features"] for row in records]
    if unique_candidates is not None and len(features) >= max(64, 4 * unique_candidates):
        n_features = _shared_feature_counts(features)
    else:
        n_features = [
            (
                np.int_(np.count_nonzero(mask))
                if type(mask) is np.ndarray and mask.dtype.kind == "b"
                else np.sum(mask)
            )
            for mask in features
        ]
    skipped = None
    if mask_limit is not None:
        invalid = np.fromiter(
            (
                row.get("skipped_invalid_mask", False) or not 1 <= count <= mask_limit
                for row, count in zip(records, n_features, strict=True)
            ),
            dtype=bool,
            count=len(records),
        )
        if invalid.any():
            skipped = invalid
            legacy_skipped = [
                i for i in np.flatnonzero(skipped) if not records[i].get("skipped_invalid_mask")
            ]
            if legacy_skipped:
                # Older checkpoints used a numeric penalty in the score columns.
                fields = ["cv_scores", *(f"test_{metric}" for metric in metrics)]
                if return_train_score:
                    fields.extend(f"train_{metric}" for metric in metrics)
                missing = dict.fromkeys(fields, np.full(len(records[0]["cv_scores"]), np.nan))
                records = list(records)
                for index in legacy_skipped:
                    records[index] = {**records[index], **missing}

    cv_results = {}
    _add_cv_statistics(
        cv_results,
        records,
        return_train_score,
        metrics,
        greater_is_better,
        unique_candidates=unique_candidates,
    )
    if skipped is not None:
        cv_results["skipped_invalid_mask"] = skipped
        for name, ranks in cv_results.items():
            if name.startswith("rank_"):
                ranks[skipped] = 1 + ranks[~skipped].max(initial=0)
    cv_results["n_features"] = n_features
    cv_results["rank_n_features"] = _rank_feature_counts(n_features)
    cv_results["features"] = features
    return cv_results
