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


def create_gasearch_cv_results_(
    logbook, space, return_train_score, metrics, greater_is_better=True
):
    cv_results = {}
    n_splits = len(logbook.chapters["parameters"].select("cv_scores")[0])

    for parameter in space.parameters:
        cv_results[f"param_{parameter}"] = logbook.chapters["parameters"].select(parameter)

    # Keys that are extended per metric in multi-metric
    for metric in metrics:
        for split in range(n_splits):
            cv_results[f"split{split}_test_{metric}"] = [
                cv_scores[split]
                for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
            ]

        cv_results[f"mean_test_{metric}"] = [
            np.mean(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
        ]
        cv_results[f"std_test_{metric}"] = [
            np.std(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
        ]

        cv_results[f"rank_test_{metric}"] = _rank_scores(
            cv_results[f"mean_test_{metric}"], greater_is_better
        )

        if return_train_score:
            for split in range(n_splits):
                cv_results[f"split{split}_train_{metric}"] = [
                    cv_scores[split]
                    for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
                ]

            cv_results[f"mean_train_{metric}"] = [
                np.mean(cv_scores)
                for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
            ]

            cv_results[f"std_train_{metric}"] = [
                np.std(cv_scores)
                for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
            ]

            cv_results[f"rank_train_{metric}"] = _rank_scores(
                cv_results[f"mean_train_{metric}"], greater_is_better
            )

    # These values are only one even with multi-metric
    cv_results["mean_fit_time"] = [
        np.nanmean(fit_time) for fit_time in logbook.chapters["parameters"].select("fit_time")
    ]
    cv_results["std_fit_time"] = [
        np.nanstd(fit_time) for fit_time in logbook.chapters["parameters"].select("fit_time")
    ]

    cv_results["mean_score_time"] = [
        np.nanmean(score_time) for score_time in logbook.chapters["parameters"].select("score_time")
    ]
    cv_results["std_score_time"] = [
        np.nanstd(score_time) for score_time in logbook.chapters["parameters"].select("score_time")
    ]

    cv_results["params"] = [
        select_dict_keys(individual, space.parameters)
        for individual in logbook.chapters["parameters"]
    ]

    return cv_results


def create_feature_selection_cv_results_(
    logbook, return_train_score, metrics, greater_is_better=True
):
    cv_results = {}
    n_splits = len(logbook.chapters["parameters"].select("cv_scores")[0])

    # Keys that are extended per metric in multi-metric
    for metric in metrics:
        for split in range(n_splits):
            cv_results[f"split{split}_test_{metric}"] = [
                cv_scores[split]
                for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
            ]

        cv_results[f"mean_test_{metric}"] = [
            np.mean(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
        ]
        cv_results[f"std_test_{metric}"] = [
            np.std(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
        ]

        cv_results[f"rank_test_{metric}"] = _rank_scores(
            cv_results[f"mean_test_{metric}"], greater_is_better
        )

        if return_train_score:
            for split in range(n_splits):
                cv_results[f"split{split}_train_{metric}"] = [
                    cv_scores[split]
                    for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
                ]

            cv_results[f"mean_train_{metric}"] = [
                np.mean(cv_scores)
                for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
            ]

            cv_results[f"std_train_{metric}"] = [
                np.std(cv_scores)
                for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
            ]

            cv_results[f"rank_train_{metric}"] = _rank_scores(
                cv_results[f"mean_train_{metric}"], greater_is_better
            )

    # These values are only one even with multi-metric
    cv_results["mean_fit_time"] = [
        np.nanmean(fit_time) for fit_time in logbook.chapters["parameters"].select("fit_time")
    ]
    cv_results["std_fit_time"] = [
        np.nanstd(fit_time) for fit_time in logbook.chapters["parameters"].select("fit_time")
    ]

    cv_results["mean_score_time"] = [
        np.nanmean(score_time) for score_time in logbook.chapters["parameters"].select("score_time")
    ]
    cv_results["std_score_time"] = [
        np.nanstd(score_time) for score_time in logbook.chapters["parameters"].select("score_time")
    ]

    cv_results["n_features"] = [
        np.sum(features) for features in logbook.chapters["parameters"].select("features")
    ]

    cv_results["rank_n_features"] = rankdata(
        np.array(cv_results["n_features"]), method="min"
    ).astype(int)

    cv_results["features"] = logbook.chapters["parameters"].select("features")

    return cv_results
