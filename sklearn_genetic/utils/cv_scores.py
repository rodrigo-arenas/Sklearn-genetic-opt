import numpy as np
from scipy.stats import rankdata


def select_dict_keys(dictionary, keys):
    return {key: dictionary[key] for key in keys}


def _rank_scores(scores, greater_is_better=True):
    """Rank candidate scores, always placing NaN scores last.

    Candidates with finite scores are ranked first; ties share the lowest rank
    (scipy's ``method="min"``). NaN scores—produced when a candidate fails under
    the default ``error_score=np.nan``—are assigned the worst ranks so they never
    outrank a valid candidate. An all-NaN input does not crash and returns
    descending ranks for every candidate.

    Parameters
    ----------
    scores : array-like
        The mean scores for each candidate. May contain NaN.
    greater_is_better : bool, default=True
        If True, higher scores rank better (e.g. accuracy). If False, lower
        scores rank better (e.g. an error/loss metric).

    Returns
    -------
    numpy.ndarray of int
        1-based ranks, where rank 1 is the best candidate and NaN scores rank last.
    """
    scores = np.asarray(scores, dtype=float)
    finite_mask = ~np.isnan(scores)

    ranks = np.empty(scores.shape[0], dtype=int)

    if finite_mask.any():
        finite_scores = scores[finite_mask]
        ordered = -finite_scores if greater_is_better else finite_scores
        ranks[finite_mask] = rankdata(ordered, method="min").astype(int)
        # NaN candidates rank strictly worse than every finite candidate.
        ranks[~finite_mask] = finite_mask.sum() + 1
    else:
        # All scores are NaN: keep a deterministic, non-crashing ranking.
        ranks[:] = 1

    return ranks


def create_gasearch_cv_results_(logbook, space, return_train_score, metrics):
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
            np.nanmean(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
        ]
        cv_results[f"std_test_{metric}"] = [
            np.nanstd(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
        ]

        cv_results[f"rank_test_{metric}"] = _rank_scores(cv_results[f"mean_test_{metric}"])

        if return_train_score:
            for split in range(n_splits):
                cv_results[f"split{split}_train_{metric}"] = [
                    cv_scores[split]
                    for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
                ]

            cv_results[f"mean_train_{metric}"] = [
                np.nanmean(cv_scores)
                for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
            ]

            cv_results[f"std_train_{metric}"] = [
                np.nanstd(cv_scores)
                for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
            ]

            cv_results[f"rank_train_{metric}"] = _rank_scores(cv_results[f"mean_train_{metric}"])

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


def create_feature_selection_cv_results_(logbook, return_train_score, metrics):
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
            np.nanmean(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
        ]
        cv_results[f"std_test_{metric}"] = [
            np.nanstd(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select(f"test_{metric}")
        ]

        cv_results[f"rank_test_{metric}"] = _rank_scores(cv_results[f"mean_test_{metric}"])

        if return_train_score:
            for split in range(n_splits):
                cv_results[f"split{split}_train_{metric}"] = [
                    cv_scores[split]
                    for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
                ]

            cv_results[f"mean_train_{metric}"] = [
                np.nanmean(cv_scores)
                for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
            ]

            cv_results[f"std_train_{metric}"] = [
                np.nanstd(cv_scores)
                for cv_scores in logbook.chapters["parameters"].select(f"train_{metric}")
            ]

            cv_results[f"rank_train_{metric}"] = _rank_scores(cv_results[f"mean_train_{metric}"])

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
