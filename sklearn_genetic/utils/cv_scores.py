import numpy as np
from scipy.stats import rankdata


def select_dict_keys(dictionary, keys):
    return {key: dictionary[key] for key in keys}


def create_gasearch_cv_results_(logbook, space, return_train_score):
    cv_results = {}
    n_splits = len(logbook.chapters["parameters"].select("cv_scores")[0])

    for parameter in space.parameters:
        cv_results[f"param_{parameter}"] = logbook.chapters["parameters"].select(
            parameter
        )

    for split in range(n_splits):
        cv_results[f"split{split}_test_score"] = [
            cv_scores[split]
            for cv_scores in logbook.chapters["parameters"].select("cv_scores")
        ]

    cv_results["mean_test_score"] = logbook.chapters["parameters"].select("score")
    cv_results["std_test_score"] = [
        np.nanstd(cv_scores)
        for cv_scores in logbook.chapters["parameters"].select("cv_scores")
    ]

    cv_results["rank_test_score"] = rankdata(
        -np.array(cv_results["mean_test_score"]), method="min"
    ).astype(int)

    if return_train_score:

        for split in range(n_splits):
            cv_results[f"split{split}_train_score"] = [
                cv_scores[split]
                for cv_scores in logbook.chapters["parameters"].select("train_score")
            ]

        cv_results["mean_train_score"] = [
            np.nanmean(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select("train_score")
        ]

        cv_results["std_train_score"] = [
            np.nanstd(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select("train_score")
        ]

        cv_results["rank_train_score"] = rankdata(
            -np.array(cv_results["mean_train_score"]), method="min"
        ).astype(int)

    cv_results["mean_fit_time"] = [
        np.nanmean(fit_time)
        for fit_time in logbook.chapters["parameters"].select("fit_time")
    ]
    cv_results["std_fit_time"] = [
        np.nanstd(fit_time)
        for fit_time in logbook.chapters["parameters"].select("fit_time")
    ]

    cv_results["mean_score_time"] = [
        np.nanmean(score_time)
        for score_time in logbook.chapters["parameters"].select("score_time")
    ]
    cv_results["std_score_time"] = [
        np.nanstd(score_time)
        for score_time in logbook.chapters["parameters"].select("score_time")
    ]

    cv_results["params"] = [
        select_dict_keys(individual, space.parameters)
        for individual in logbook.chapters["parameters"]
    ]

    return cv_results


def create_feature_selection_cv_results_(logbook, return_train_score):
    cv_results = {}
    n_splits = len(logbook.chapters["parameters"].select("cv_scores")[0])

    for split in range(n_splits):
        cv_results[f"split{split}_test_score"] = [
            cv_scores[split]
            for cv_scores in logbook.chapters["parameters"].select("cv_scores")
        ]

    cv_results["mean_test_score"] = logbook.chapters["parameters"].select("score")
    cv_results["std_test_score"] = [
        np.nanstd(cv_scores)
        for cv_scores in logbook.chapters["parameters"].select("cv_scores")
    ]

    cv_results["rank_test_score"] = rankdata(
        -np.array(cv_results["mean_test_score"]), method="min"
    ).astype(int)

    if return_train_score:

        for split in range(n_splits):
            cv_results[f"split{split}_train_score"] = [
                cv_scores[split]
                for cv_scores in logbook.chapters["parameters"].select("train_score")
            ]

        cv_results["mean_train_score"] = [
            np.nanmean(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select("train_score")
        ]

        cv_results["std_train_score"] = [
            np.nanstd(cv_scores)
            for cv_scores in logbook.chapters["parameters"].select("train_score")
        ]

        cv_results["rank_train_score"] = rankdata(
            -np.array(cv_results["mean_train_score"]), method="min"
        ).astype(int)

    cv_results["mean_fit_time"] = [
        np.nanmean(fit_time)
        for fit_time in logbook.chapters["parameters"].select("fit_time")
    ]
    cv_results["std_fit_time"] = [
        np.nanstd(fit_time)
        for fit_time in logbook.chapters["parameters"].select("fit_time")
    ]

    cv_results["mean_score_time"] = [
        np.nanmean(score_time)
        for score_time in logbook.chapters["parameters"].select("score_time")
    ]
    cv_results["std_score_time"] = [
        np.nanstd(score_time)
        for score_time in logbook.chapters["parameters"].select("score_time")
    ]

    cv_results["n_features"] = [
        np.sum(features)
        for features in logbook.chapters["parameters"].select("features")
    ]

    cv_results["rank_n_features"] = rankdata(
        np.array(cv_results["n_features"]), method="min"
    ).astype(int)

    cv_results["features"] = logbook.chapters["parameters"].select("features")

    return cv_results
