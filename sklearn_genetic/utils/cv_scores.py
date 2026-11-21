import numpy as np
from numbers import Real
from scipy.stats import rankdata
from sklearn.utils import check_random_state, resample


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


def _rank_scores(scores, greater_is_better=True):
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
        ranks[finite_mask] = rankdata(ordered, method="min").astype(int)
        # Non-finite candidates rank strictly worse than every finite candidate.
        ranks[~finite_mask] = finite_mask.sum() + 1
    else:
        # All scores are non-finite: keep a deterministic, non-crashing ranking.
        ranks[:] = 1

    return ranks


def _add_cv_statistics(cv_results, records, return_train_score, metrics, greater_is_better):
    n_splits = len(records[0]["cv_scores"])
    for metric in metrics:
        for subset in ("test", "train") if return_train_score else ("test",):
            name = f"{subset}_{metric}"
            scores = [row[name] for row in records]
            for split in range(n_splits):
                cv_results[f"split{split}_{name}"] = [row[split] for row in scores]
            means = [np.mean(row) for row in scores]
            cv_results[f"mean_{name}"] = means
            cv_results[f"std_{name}"] = [np.std(row) for row in scores]
            cv_results[f"rank_{name}"] = _rank_scores(means, greater_is_better)

    for name in ("fit_time", "score_time"):
        rows = [row[name] for row in records]
        means, deviations = [], []
        # Bound temporary timing matrices even for long searches and leave-one-out CV.
        batch_size = max(1, 65536 // max(1, n_splits))
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            first = batch[0]
            if all(
                type(row) is np.ndarray
                and row.ndim == 1
                and row.dtype.kind == "f"
                and row.shape == (n_splits,)
                and row.dtype == first.dtype
                for row in batch
            ):
                times = np.asarray(batch)
                means.extend(np.nanmean(times, axis=1))
                deviations.extend(np.nanstd(times, axis=1))
            else:
                # Keep per-row semantics for heterogeneous or legacy logbooks.
                means.extend(np.nanmean(row) for row in batch)
                deviations.extend(np.nanstd(row) for row in batch)
        cv_results[f"mean_{name}"] = means
        cv_results[f"std_{name}"] = deviations


def create_gasearch_cv_results_(
    records, space, return_train_score, metrics, greater_is_better=True
):
    cv_results = {f"param_{name}": [row[name] for row in records] for name in space.parameters}
    _add_cv_statistics(cv_results, records, return_train_score, metrics, greater_is_better)
    cv_results["params"] = [
        select_dict_keys(individual, space.parameters) for individual in records
    ]
    return cv_results


def create_feature_selection_cv_results_(
    records, return_train_score, metrics, greater_is_better=True
):
    cv_results = {}
    _add_cv_statistics(cv_results, records, return_train_score, metrics, greater_is_better)
    features = [row["features"] for row in records]
    cv_results["n_features"] = [np.sum(mask) for mask in features]
    cv_results["rank_n_features"] = rankdata(cv_results["n_features"], method="min").astype(int)
    cv_results["features"] = features
    return cv_results
