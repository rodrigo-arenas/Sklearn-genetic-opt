import numpy as np
from scipy.stats import rankdata


def select_dict_keys(dictionary, keys):
    return {key: dictionary[key] for key in keys}


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

        cv_results[f"rank_test_{metric}"] = rankdata(
            -np.array(cv_results[f"mean_test_{metric}"]), method="min"
        ).astype(int)

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

            cv_results[f"rank_train_{metric}"] = rankdata(
                -np.array(cv_results[f"mean_train_{metric}"]), method="min"
            ).astype(int)

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

        cv_results[f"rank_test_{metric}"] = rankdata(
            -np.array(cv_results[f"mean_test_{metric}"]), method="min"
        ).astype(int)

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

            cv_results[f"rank_train_{metric}"] = rankdata(
                -np.array(cv_results[f"mean_train_{metric}"]), method="min"
            ).astype(int)

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
