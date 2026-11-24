from collections.abc import Iterable
from copy import deepcopy
import inspect
from math import isnan
from numbers import Integral
import random
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.model_selection import check_cv, cross_validate
from sklearn.base import is_classifier, is_regressor, BaseEstimator, MetaEstimatorMixin
from sklearn.pipeline import Pipeline

try:
    from sklearn.base import is_outlier_detector
except ImportError:
    # Fallback for older sklearn versions
    def is_outlier_detector(estimator):
        return hasattr(estimator, "fit_predict") and hasattr(estimator, "decision_function")


try:
    from sklearn.utils.validation import _check_feature_names
except ImportError:

    def _check_feature_names(estimator, X, *, reset):
        estimator._check_feature_names(X, reset=reset)


def _safe_estimator_check(check, estimator):
    try:
        return check(estimator)
    except AttributeError:
        return False


def _is_classifier(estimator):
    return _safe_estimator_check(is_classifier, estimator)


def _is_regressor(estimator):
    return _safe_estimator_check(is_regressor, estimator)


def _is_outlier_detector(estimator):
    return _safe_estimator_check(is_outlier_detector, estimator)


_CONTINUOUS_OUTLIER_RESPONSES = ("score_samples", "decision_function")


def _outlier_response(estimator):
    for name in (*_CONTINUOUS_OUTLIER_RESPONSES, "predict"):
        response = getattr(estimator, name, None)
        if callable(response):
            return name, response
    raise ValueError(
        f"Default outlier scoring requires {type(estimator).__name__} to expose "
        "score_samples, decision_function, or predict for unseen data. "
        "For LocalOutlierFactor, set novelty=True. You can also supply a "
        "scoring callable that evaluates the fitted detector."
    )


def _default_outlier_scorer(estimator, X, y=None, sample_weight=None):
    name, response = _outlier_response(estimator)
    scores = response(X)
    if name == "predict":
        scores = np.asarray(scores) == 1
    return np.mean(scores) if sample_weight is None else np.average(scores, weights=sample_weight)


from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    _check_y,
    _num_samples,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)
from sklearn.feature_selection._from_model import _estimator_has
from sklearn.metrics import check_scoring
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics._scorer import _MultimetricScorer, _check_multimetric_scoring

try:
    from sklearn.utils import get_tags
    from sklearn.utils.validation import validate_data
except ImportError:  # scikit-learn 1.5
    from sklearn.utils._tags import _safe_tags

    validate_data = None
    _FINITE_CHECK = "force_all_finite"
else:
    _FINITE_CHECK = "ensure_all_finite"

from .parameters import Criteria
from .space import Categorical, Continuous, Integer, Space
from ._base import GeneticEstimatorMixin, generation_history, reset_adapters as _reset_adapters
from .backends import _validate_search_counts
from .callbacks.validations import check_callback
from .utils.cv_scores import (
    create_gasearch_cv_results_,
    create_feature_selection_cv_results_,
)
from .utils._score_statistics import score_mean
from .evaluation import (
    candidate_parameters,
    candidate_records,
    create_fit_stats as _create_fit_stats,
    evaluate_population as _evaluate_population_batch,
    parameter_record_conflicts,
    preserve_candidate_parameters,
    record_candidate,
    record_fit_stats as _record_fit_stats,
    validate_error_score as _validate_error_score,
    validate_parallel_backend as _validate_parallel_backend,
)
from .population import (
    initialize_feature_population,
    initialize_search_population,
    validate_population_initializer as _validate_population_initializer,
)
from .optimizer_control import (
    mutable_dimensions,
    validate_optimizer_control as _validate_optimizer_control,
)

from .config import EvolutionConfig, OptimizationConfig, PopulationConfig, RuntimeConfig


def _allows_nan(estimator):
    if validate_data is None:
        return _safe_tags(estimator, key="allow_nan")
    return get_tags(estimator).input_tags.allow_nan


def _requires_target(estimator):
    if validate_data is None:
        required = _safe_tags(estimator, key="requires_y")
    else:
        required = get_tags(estimator).target_tags.required
    if required:
        return True
    # Pipeline does not propagate the target requirement of its steps.
    # Subclasses can define their own fit contract through their tags.
    if type(estimator) is Pipeline:
        return any(
            _requires_target(step)
            for _, step in estimator.steps
            if step is not None and not isinstance(step, str)
        )
    return False


def _check_required_target(estimator, y):
    if y is None and _requires_target(estimator):
        raise ValueError(
            f"{type(estimator).__name__} requires y to be passed, but the target y is None."
        )


def _feature_target_estimator(estimator):
    # Standard pipelines forward y to the final estimator. Older versions
    # omit some or all of its target shape tags on the pipeline itself.
    while type(estimator) is Pipeline and estimator.steps:
        estimator = estimator.steps[-1][1]
    return estimator


def _feature_target_format(estimator):
    estimator = _feature_target_estimator(estimator)
    if validate_data is None:
        tags = _safe_tags(estimator)
        matrix_required = tags["multioutput_only"]
        accepts_matrix = matrix_required or tags["multioutput"] or tags["multilabel"]
    else:
        tags = get_tags(estimator)
        target = tags.target_tags
        matrix_required = not target.single_output or target.two_d_labels
        multilabel = tags.classifier_tags is not None and tags.classifier_tags.multi_label
        accepts_matrix = matrix_required or target.multi_output or multilabel
    return accepts_matrix, matrix_required


def _feature_input_estimator(estimator):
    """Find the first estimator that receives the raw feature columns."""
    if type(estimator) is not Pipeline:
        return estimator
    for _, step in estimator.steps:
        if step is None or isinstance(step, str):
            continue
        first = _feature_input_estimator(step)
        if first is not None:
            return first
    return None


def _check_feature_selection_input(estimator):
    first = _feature_input_estimator(estimator)
    if first is None:
        return
    if validate_data is None:
        pairwise = _safe_tags(first, key="pairwise")
    else:
        pairwise = get_tags(first).input_tags.pairwise
    if pairwise:
        raise ValueError(
            "GAFeatureSelectionCV requires original feature columns. "
            f"{type(first).__name__} expects a precomputed pairwise matrix "
            "whose columns represent samples."
        )


def _feature_input_checks(estimator):
    if validate_data is None:
        tags = _safe_tags(estimator)
        preserve_dtype = "categorical" in tags["X_types"] or "string" in tags["X_types"]
        allow_nan = tags["allow_nan"]
    else:
        inputs = get_tags(estimator).input_tags
        preserve_dtype = inputs.categorical or inputs.string
        allow_nan = inputs.allow_nan
    return {
        "dtype": None if preserve_dtype else "numeric",
        _FINITE_CHECK: "allow-nan" if allow_nan else True,
    }


def _check_feature_dataframe(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("preserve_dataframe=True requires X to be a pandas DataFrame")
    if not all(X.shape):
        raise ValueError("X must contain at least one sample and one feature")


def _get_criteria_sign(criteria):
    if criteria not in Criteria.list():
        raise ValueError(f"Criteria must be one of {Criteria.list()}, got {criteria} instead")
    return 1.0 if criteria == Criteria.max.value else -1.0


def _restore_search_rngs(state):
    if state is not None:
        python_state, numpy_state = state
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _seed_global_rngs(random_state):
    """Seed the global ``random`` and NumPy RNGs from a single ``random_state``.

    A genetic search draws on several sources of randomness — population
    initialization, DEAP's mutation/crossover operators (which use the global
    ``random`` module), random immigrants, and NumPy sampling. Seeding them all
    from one estimator-level ``random_state`` at the start of ``fit`` gives
    reproducible runs without callers having to seed the global RNGs by hand.

    ``random_state=None`` keeps the default non-deterministic behaviour, matching
    scikit-learn's convention.
    """
    if random_state is None:
        return
    seed = int(check_random_state(random_state).randint(0, 2**31 - 1))
    random.seed(seed)
    np.random.seed(seed)


def _resolve_config_value(config, field_name, fallback):
    if config is None:
        return fallback

    return getattr(config, field_name, fallback)


_IMMUTABLE_PARAMETER_TYPES = {type(None), bool, int, float, complex, str, bytes}


def _clone_mutable_candidate_params(params):
    # Keyword expansion copies the mapping. Only exact immutable scalars can
    # safely bypass the copy of each value.
    if type(params) is not dict or any(
        type(value) not in _IMMUTABLE_PARAMETER_TYPES for value in params.values()
    ):
        return clone(params, safe=False)
    return params


def _evolution_score(score, criteria_sign):
    # DEAP does not order NaN reliably, and infinite scores do not describe
    # usable fitness. Keep raw CV scores visible, but rank these candidates last.
    if not np.isfinite(score):
        return -np.inf if criteria_sign > 0 else np.inf
    return score


def _normalize_cached_fitness(cache, criteria_sign):
    # Checkpoints made before non-finite fitness was penalized can still hold
    # raw NaN/Inf values. Convert only the evolutionary fitness on restore;
    # their recorded CV scores remain available in the candidate logbook.
    for result in cache.values():
        fitness = result["fitness"]
        if not np.isfinite(fitness[0]):
            normalized = _evolution_score(fitness[0], criteria_sign)
            result["fitness"] = type(fitness)((normalized, *fitness[1:]))
    return cache


def _restore_legacy_checkpoint(estimator, checkpoint):
    if checkpoint is None:
        return {}
    # Older DEAP files contain only the constructor state and generation log.
    runtime = checkpoint.get("runtime_state") or {}
    requested_metric = (
        estimator.refit if isinstance(estimator.refit, str) else estimator.objective_metric
    )
    if (
        "refit_metric" in runtime
        and isinstance(requested_metric, str)
        and requested_metric != runtime["refit_metric"]
    ):
        raise ValueError("Checkpoint optimization metric differs from this fit")
    state = checkpoint["estimator_state"]
    estimator.__dict__.update(state)
    if "final_selection_cv" in state:
        estimator._explicit_finalist_cv = state["final_selection_cv"]
    if runtime.get("fitness_cache") is not None:
        estimator.fitness_cache = _normalize_cached_fitness(
            runtime["fitness_cache"], _get_criteria_sign(estimator.criteria)
        )
    return runtime


def _check_fit_params_supported(estimator, fit_params):
    """Validate keyword arguments forwarded directly to fit without routing.

    ``fit_params`` (e.g. ``sample_weight``) are forwarded to the estimator's
    ``fit`` during cross-validation and refit. If the estimator doesn't accept a
    given key it would otherwise blow up mid-search with scikit-learn's generic
    "unexpected keyword argument" error, so we check up front and name the
    offending parameter and estimator instead.

    The check is intentionally conservative: if ``fit`` takes ``**kwargs`` (as
    Pipeline and many meta-estimators do) we can't tell statically which keys are
    valid, so we skip it and let the estimator decide at fit time. That keeps
    valid routed usage like ``clf__sample_weight`` working untouched.
    """
    if not fit_params or _routing_enabled():
        return

    try:
        parameters = inspect.signature(estimator.fit).parameters.values()
    except (TypeError, ValueError):
        # Can't introspect fit (e.g. a C-level callable); leave it to fit time.
        return

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
        return

    accepted = {
        p.name
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unsupported = sorted(name for name in fit_params if name not in accepted)
    if unsupported:
        raise TypeError(
            f"{type(estimator).__name__}.fit does not accept the metadata "
            f"{unsupported}; it was passed to fit() but the wrapped estimator "
            f"cannot use it. Remove it or use an estimator whose fit accepts it."
        )


def _split_accepts_groups(split):
    """Return whether the callable accepts groups as a keyword, or None if unknown."""
    if not callable(split):
        return False
    try:
        signature = inspect.signature(split)
    except (TypeError, ValueError):
        return None
    try:
        signature.bind_partial(groups=None)
    except TypeError:
        return False
    return True


_FINALIST_PREFIX_MIN_ROWS = 1024


def _can_combine_fold_scores(values):
    # Combining arrays can promote large integers to float. A NaN already
    # represents an undefined score and can be carried over.
    return all(abs(value) < 2**53 or isnan(value) for value in map(float, values))


def _cv_split_count(splits, parameter):
    count = len(splits)
    if count == 0:
        raise ValueError(f"{parameter} must provide at least one train/test split.")
    return count


def _validate_explicit_folds(folds, n_samples, *, parameter):
    """Check materialized folds and their ordinary row selectors."""
    for number, fold in enumerate(folds, start=1):
        if type(fold) not in (list, tuple, np.ndarray):
            continue
        label = f"{parameter} fold {number}"
        size = 0 if type(fold) is np.ndarray and fold.ndim == 0 else len(fold)
        if size != 2:
            raise ValueError(
                f"{label} must contain exactly two selectors (training and validation), got {size}"
            )
        for partition, rows in zip(("training", "validation"), fold, strict=True):
            if type(rows) not in (list, tuple, np.ndarray):
                continue
            try:
                rows = np.asarray(rows)
            except (TypeError, ValueError):
                continue
            if rows.ndim != 1:
                continue
            if rows.dtype.kind == "b" and rows.size != n_samples:
                raise IndexError(
                    f"{label} {partition} mask must have {n_samples} entries, got {rows.size}"
                )
            if (
                rows.dtype.kind in "iu"
                and rows.size
                and (int(rows.min()) < -n_samples or int(rows.max()) >= n_samples)
            ):
                raise IndexError(
                    f"{label} {partition} indices must be between {-n_samples} and {n_samples - 1}"
                )


def _matching_row_indices(indices, n_samples):
    if indices.ndim != 1 or n_samples is None:
        return indices
    if indices.dtype.kind == "b":
        return np.flatnonzero(indices) if indices.size == n_samples else indices
    if indices.dtype.kind == "i" and indices.size:
        smallest = int(indices.min())
        if -n_samples <= smallest < 0 and int(indices.max()) < n_samples:
            # Widen before adding the row count to narrow signed indices.
            indices = indices.astype(np.intp, copy=True)
            indices[indices < 0] += n_samples
    return indices


class _FinalSelectionMixin:
    """Shared finalist ranking, validation and selection for both search types."""

    def _configured_scorer(self):
        if callable(self.scoring):
            return self.scoring, False
        if self.scoring is None and _is_outlier_detector(self.estimator):
            return _default_outlier_scorer, False
        if self.scoring is None or isinstance(self.scoring, str):
            return check_scoring(self.estimator, self.scoring), False
        return _check_multimetric_scoring(self.estimator, self.scoring), True

    def _set_search_scoring(self, checkpoint):
        self.scorer_, self.multimetric_ = self._configured_scorer()
        if self.multimetric_:
            self.metrics_list = list(self.scorer_)
            self.refit_metric = self._multimetric_objective()
        else:
            if self.objective_metric is not None:
                raise ValueError(
                    "objective_metric requires multiple-metric scoring. "
                    "Pass a dictionary of named scorers to scoring."
                )
            self.refit_metric = "score"
            self.metrics_list = [self.refit_metric]
            if self.scoring is None and self.scorer_ is _default_outlier_scorer:
                if not hasattr(self, "space"):
                    _outlier_response(self.estimator)
                self._check_outlier_checkpoint(checkpoint)

    def _set_search_history(self, log):
        self.history = generation_history(log)
        if self.cv_results_["all_fits_failed"].all():
            raise ValueError(
                "All search candidates failed to fit. Set error_score='raise' to debug."
            )
        scores = self.cv_results_[f"mean_test_{self.refit_metric}"]
        if not any(map(np.isfinite, scores)):
            metric = self.scoring if isinstance(self.scoring, str) else self.refit_metric
            warnings.warn(
                "No candidate has a finite mean CV score for the objective metric "
                f"{metric!r}. Inspect the split scores in cv_results_.",
                UserWarning,
                stacklevel=3,
            )

    def _check_outlier_checkpoint(self, checkpoint):
        if checkpoint is None or checkpoint.get("outlier_scoring_uses_fitted_detector"):
            return

        def used_fitted_detector(estimator):
            return any(
                callable(getattr(estimator, method, None))
                for method in _CONTINUOUS_OUTLIER_RESPONSES
            )

        if hasattr(self, "space"):
            records = checkpoint.get("candidate_records")
            if records is None:
                runtime = checkpoint.get("runtime_state") or {}
                book = runtime.get("candidate_logbook")
                records = book.chapters["parameters"] if book is not None else ()
            names = self.space.parameters

            def historical_candidate(row):
                values = candidate_parameters(row)
                params = {name: values[name] for name in names}
                return clone(self.estimator).set_params(**_clone_mutable_candidate_params(params))

            compatible = records and all(
                used_fitted_detector(historical_candidate(row)) for row in records
            )
        else:
            compatible = used_fitted_detector(self.estimator)
        if not compatible:
            raise ValueError(
                "This checkpoint may contain outlier scores computed by fitting "
                "validation rows. Use a new checkpoint path to start a fresh search."
            )

    def _multimetric_objective(self):
        if self.refit is False or callable(self.refit):
            metric = self.objective_metric
            if isinstance(metric, str) and metric in self.scorer_:
                return metric
            mode = "callable refit" if callable(self.refit) else "refit=False"
            raise ValueError(
                f"With multiple-metric scoring and {mode}, set objective_metric "
                f"to one of {self.metrics_list!r}"
            )
        if isinstance(self.refit, str) and self.refit in self.scorer_:
            metric = self.objective_metric
            if metric is not None and (not isinstance(metric, str) or metric != self.refit):
                raise ValueError("objective_metric must match the named refit metric")
            return self.refit
        raise ValueError(
            "With multiple-metric scoring, use a scorer name for refit, or set "
            "objective_metric with callable refit or refit=False"
        )

    def _validate_final_selection_top_k(self):
        if isinstance(self.final_selection_top_k, (bool, np.bool_)) or not isinstance(
            self.final_selection_top_k, Integral
        ):
            raise ValueError("final_selection_top_k must be a positive integer")
        if self.final_selection_top_k < 1:
            raise ValueError("final_selection_top_k must be greater than or equal to 1")

    def _candidate_params_from_index(self, index):
        if hasattr(self, "space"):
            return self.cv_results_["params"][index]
        return {"features": tuple(bool(value) for value in self.cv_results_["features"][index])}

    def _best_ranked_index(self):
        return int(np.asarray(self.cv_results_[f"rank_test_{self.refit_metric}"]).argmin())

    def _finalist_key(self, index):
        if hasattr(self, "space"):
            params = self.cv_results_["params"][index]
            return self._individual_key(params[name] for name in self.space.parameters)
        mask = np.asarray(self.cv_results_["features"][index], dtype=bool)
        return mask.size, np.packbits(mask).tobytes()

    def _mask_finalist_keys(self, indices):
        if (
            len(indices) < 128
            or hasattr(self, "space")
            or getattr(self._finalist_key, "__func__", None)
            is not _FinalSelectionMixin._finalist_key
        ):
            yield from ((index, self._finalist_key(index)) for index in indices)
            return

        features = self.cv_results_["features"]
        first = features[indices[0]]
        # Wider masks leave too few rows per batch to amortize the copy.
        if (
            type(first) is not np.ndarray
            or first.ndim != 1
            or first.dtype != bool
            or not 0 < first.size <= 4096
        ):
            yield from ((index, self._finalist_key(index)) for index in indices)
            return
        block_size = 262144 // first.size
        for start in range(0, len(indices), block_size):
            block = indices[start : start + block_size]
            masks = [features[index] for index in block]
            if all(
                type(mask) is np.ndarray and mask.dtype == bool and mask.shape == first.shape
                for mask in masks
            ):
                packed = np.packbits(masks, axis=1)
                yield from (
                    (index, (first.size, bits.tobytes()))
                    for index, bits in zip(block, packed, strict=True)
                )
            else:
                yield from ((index, self._finalist_key(index)) for index in block)

    def _top_candidate_indices(self):
        ranks = np.asarray(self.cv_results_[f"rank_test_{self.refit_metric}"])
        counts = self.cv_results_.get("n_features")
        limit = int(self.final_selection_top_k)
        rows = None
        flat = False
        if ranks.size >= _FINALIST_PREFIX_MIN_ROWS:
            flat = ranks[0] == ranks[-1] and np.all(ranks == ranks[0])
        if not flat and ranks.size >= _FINALIST_PREFIX_MIN_ROWS and 8 * limit < ranks.size:
            unique = getattr(self, "fit_stats_", {}).get("unique_candidates", ranks.size)
            if 2 * unique >= ranks.size:
                # Keep the complete cutoff rank, including ties. A short prefix
                # often contains enough distinct candidates to avoid a full sort.
                eligible = ranks <= 4 * limit
                if limit <= np.count_nonzero(eligible) <= ranks.size // 2:
                    rows = np.flatnonzero(eligible)
        indices = []
        seen = set()
        start = 0
        while True:
            eligible_ranks = ranks if rows is None else ranks[rows]
            if flat:
                order = range(ranks.size) if counts is None else np.argsort(counts, kind="stable")
            elif counts is None:
                order = np.argsort(eligible_ranks, kind="stable")
            else:
                eligible_counts = counts if rows is None else np.asarray(counts)[rows]
                order = np.lexsort((eligible_counts, eligible_ranks))
            candidates = order if rows is None else rows[order]
            for index in candidates[start:]:
                key = self._finalist_key(index)
                if key not in seen:
                    seen.add(key)
                    indices.append(index)
                    if len(indices) == limit:
                        break
            if rows is None or len(indices) == limit:
                break
            # Every prefix rank is complete. A full sort can resume after it
            # with the same selected indices and duplicate keys.
            start = len(rows)
            rows = None
        if counts is not None and len(indices) > 1 and ranks[indices[-2]] == ranks[indices[-1]]:
            # A flat proxy must not reserve every tied slot for sparse masks.
            # Keep the smallest finalist and include a denser mask at the same
            # cutoff rank, without displacing a better score or adding CV calls.
            cutoff = indices[-1]
            if flat:
                tied = candidates
            else:
                start = eligible_ranks.searchsorted(ranks[cutoff], side="left", sorter=order)
                end = eligible_ranks.searchsorted(ranks[cutoff], side="right", sorter=order)
                tied = candidates[start:end]
            if len(tied) >= 128:
                sizes = np.asarray(counts)
                denser = tied[sizes[tied] > sizes[cutoff]]
            else:
                denser = [index for index in tied if counts[index] > counts[cutoff]]
            for index, key in self._mask_finalist_keys(denser):
                if key not in seen:
                    # A repeated row must not displace a distinct mask first
                    # observed later at the same density and score.
                    seen.add(key)
                    indices[-1] = index
        return indices

    def _final_selection_splits(self):
        if self.final_selection_cv is None:
            return self._cv_splits

        splits = self._materialize_cv_splits(
            self.final_selection_cv, classifier=_is_classifier(self.estimator), other_cv=self.cv
        )
        _cv_split_count(splits, "final_selection_cv")
        return splits

    def _materialize_cv_splits(self, cv, *, classifier, copy_indices=True, other_cv=None):
        explicit = cv is not None and not isinstance(cv, Integral)
        if (
            copy_indices
            and explicit
            and isinstance(cv, Iterable)
            and not isinstance(cv, str)
            and not hasattr(cv, "split")
        ):
            # check_cv consumes a raw fold iterable before its wrapper can yield it.
            folds = cv
        else:
            split = check_cv(cv, self.y_, classifier=classifier).split
            groups = getattr(self, "groups_", None)
            groups_kwargs = {"groups": groups} if explicit and groups is not None else {}
            if (
                groups_kwargs
                and other_cv is not None
                and _split_accepts_groups(split) is False
                and _split_accepts_groups(getattr(other_cv, "split", None)) is True
            ):
                groups_kwargs = {}
            folds = split(self.X_, self.y_, **groups_kwargs)
        # A caller-provided splitter can overwrite the previous split on next().
        # The splitters created from int/None already yield independent arrays.
        return list(map(deepcopy, folds)) if copy_indices and explicit else list(folds)

    def _score_final_candidate(self, params, cv_splits):
        local_estimator = self.estimator
        if hasattr(self, "space"):
            local_estimator = clone(local_estimator)
            local_estimator.set_params(**_clone_mutable_candidate_params(params))
            X = self.X_
        else:
            # As in feature search, cross_validate clones the template for each fold.
            X = self._select_feature_columns(np.asarray(params["features"], dtype=bool))

        cv_results = self._cross_validate_candidate(
            local_estimator,
            X,
            cv_splits,
            n_jobs=self.n_jobs,
            return_train_score=False,
        )
        self.fit_counts_["finalist_cv_calls"] += 1
        self.fit_counts_["finalist_fold_fits"] += len(cv_splits)
        cv_scores = cv_results[f"test_{self.refit_metric}"]
        return float(score_mean(cv_scores)), cv_scores

    def _constant_cv_scores(self, score, n_splits, return_train_score):
        subsets = ("test", "train") if return_train_score else ("test",)
        return {
            f"{subset}_{metric}": np.full(n_splits, score)
            for metric in self.metrics_list
            for subset in subsets
        }

    def _cross_validate_candidate(self, estimator, X, splits, *, n_jobs, return_train_score):
        try:
            cv_results = cross_validate(
                estimator,
                X,
                self.y_,
                cv=splits,
                scoring=self.scorer_,
                n_jobs=n_jobs,
                pre_dispatch=self.pre_dispatch,
                error_score=self.error_score,
                return_train_score=return_train_score,
                params=getattr(self, "_fit_params", None) or None,
            )
        except ValueError as error:
            # cross_validate raises an aggregate error when every fold fails to fit.
            # Leave other errors (e.g. invalid metadata routing) to the caller.
            n_splits = len(splits)
            prefix = (
                f"\nAll the {n_splits} fits failed.\n"
                "It is very likely that your model is misconfigured.\n"
            )
            if self.error_score == "raise" or not str(error).startswith(prefix):
                raise
            warnings.warn(str(error).strip(), FitFailedWarning, stacklevel=2)
            return {
                "fit_time": None,
                "score_time": np.zeros(n_splits),
                **self._constant_cv_scores(np.nan, n_splits, return_train_score),
            }
        if callable(self.scoring):
            test_columns = {column for column in cv_results if column.startswith("test_")}
            if test_columns != {"test_score"}:
                raise ValueError(
                    "A callable passed as scoring must return a single numeric score per fold. "
                    "Pass a dictionary of named scorers to scoring for multiple metrics."
                )
        return cv_results

    def _fit_metadata_router(self, estimator, scorer):
        if isinstance(scorer, dict):
            scorer = _MultimetricScorer(scorers=scorer)
        return (
            MetadataRouter(owner=type(self).__name__)
            .add(
                estimator=estimator,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit_transform", callee="fit"),
            )
            .add(
                scorer=scorer,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="score")
                .add(caller="fit_transform", callee="score"),
            )
        )

    def _refit_metadata(self, estimator):
        params = getattr(self, "_fit_params", {})
        if not params or not _routing_enabled():
            return params
        router = self._fit_metadata_router(estimator, self.scorer_)
        return process_routing(router, "fit", **params).estimator.fit

    def _target_args(self, y):
        if y is None and getattr(self, "y_", ()) is None:
            return ()
        return (y,)

    def _set_search_splits(self, checkpoint, optimizer, runtime):
        outlier = _is_outlier_detector(self.estimator)
        self._copy_cv_indices = checkpoint is None or runtime.get("copy_cv_indices", False)
        self._save_outlier_cv_splits = (
            outlier and self.cv is not None and not isinstance(self.cv, int)
        )
        if checkpoint is not None and optimizer is not None:
            splits = checkpoint["cv_splits"]
        elif "search_cv_splits" in runtime:
            splits = runtime["search_cv_splits"]
        elif "outlier_cv_splits" in runtime:
            splits = runtime["outlier_cv_splits"]
        else:
            cv = self.cv
            if checkpoint is not None and self._save_outlier_cv_splits:
                # Historical DEAP outlier searches used five-fold KFold for
                # explicit CV. Continue with the folds that produced the scores.
                cv = None
            splits = self._materialize_cv_splits(
                cv,
                classifier=not outlier and _is_classifier(self.estimator),
                copy_indices=self._copy_cv_indices,
                other_cv=self.final_selection_cv if self.refit and self.final_selection else None,
            )
            if type(cv) in (list, tuple):
                _validate_explicit_folds(splits, _num_samples(self.X_), parameter="cv")
            # Keep the historical fold materialization for old DEAP checkpoints,
            # whose rows may already contain scores from aliased indices.
            if not self._copy_cv_indices and self.final_selection == "reuse_folds":
                splits = deepcopy(splits)
        self.n_splits_ = _cv_split_count(splits, "cv")
        self._cv_splits = splits

    @staticmethod
    def _same_fold(left, right, n_samples):
        left_train, left_test = left
        right_train, right_test = right
        for first, second in ((left_train, right_train), (left_test, right_test)):
            first, second = np.asarray(first), np.asarray(second)
            if (
                first.ndim != 1
                or second.ndim != 1
                or first.dtype.kind not in "biu"
                or second.dtype.kind not in "biu"
            ):
                return False
            first_is_mask = first.dtype.kind == "b"
            if first_is_mask == (second.dtype.kind == "b"):
                if first is second or np.array_equal(first, second):
                    continue
                if first_is_mask:
                    return False
            if n_samples is None:
                return False
            first = _matching_row_indices(first, n_samples)
            second = _matching_row_indices(second, n_samples)
            if (first.dtype.kind == "b") != (second.dtype.kind == "b") or not np.array_equal(
                first, second
            ):
                return False
        return True

    @staticmethod
    def _test_indices_key(fold, n_samples):
        _, test = fold
        test = np.asarray(test)
        if test.ndim != 1 or test.dtype.kind not in "biu":
            return None
        if test.dtype.kind == "b":
            test = _matching_row_indices(test, n_samples)
        key = (test.dtype.kind == "b", test.size)
        if not test.size:
            return key
        first, last = int(test[0]), int(test[-1])
        if n_samples is not None:
            if -n_samples <= first < 0:
                first += n_samples
            if -n_samples <= last < 0:
                last += n_samples
        return (*key, first, last)

    def _matching_search_folds(self, final_splits):
        search_splits = self._cv_splits
        if final_splits is search_splits:
            return {
                index: index
                for index, fold in enumerate(search_splits)
                if self._same_fold(fold, fold, None)
            }

        # If the fold sequence is unchanged, retain each score at its original
        # position. Identical selectors can occur more than once in a sequence.
        n_samples = _num_samples(self.X_)
        if len(final_splits) == len(search_splits) and all(
            self._same_fold(final, search, n_samples)
            for final, search in zip(final_splits, search_splits, strict=True)
        ):
            return {index: index for index in range(len(search_splits))}

        # Across different sequences, the key narrows the candidates. Verify
        # the ordered training and validation indices before reusing a score.
        sources = {}
        for index, fold in enumerate(search_splits):
            key = self._test_indices_key(fold, n_samples)
            if key is not None:
                sources.setdefault(key, []).append(index)

        reused = {}
        for position, fold in enumerate(final_splits):
            key = self._test_indices_key(fold, n_samples)
            for source in sources.get(key, ()):
                if self._same_fold(fold, search_splits[source], n_samples):
                    reused[position] = source
                    break
        return reused

    def _score_final_folds(self, index, params, splits, reused, missing):
        if (
            reused
            and len(reused) == len(self._cv_splits) == len(splits)
            and all(position == source for position, source in reused.items())
        ):
            # The same aggregation would preserve the search array's dtype,
            # including any promotions that occurred across folds.
            values = np.asarray(candidate_records(self)[index][f"test_{self.refit_metric}"])
            return float(score_mean(values)), values, len(splits)

        scores = [None] * len(splits)
        for position, source in reused.items():
            scores[position] = self.cv_results_[f"split{source}_test_{self.refit_metric}"][index]

        fresh = ()
        if reused and _can_combine_fold_scores(scores[i] for i in reused):
            fresh = self._score_final_candidate(params, missing)[1] if missing else ()
            if _can_combine_fold_scores(fresh):
                remaining = iter(fresh)
                ordered = np.asarray(
                    [next(remaining) if score is None else score for score in scores]
                )
                return float(score_mean(ordered)), ordered, len(reused)
        if reused and (not missing or (len(fresh) and np.asarray(fresh).dtype.kind in "iu")):
            values = np.asarray(candidate_records(self)[index][f"test_{self.refit_metric}"])
            if values.dtype.kind in "iu":
                # Infer the final dtype from exact Python integers, as CV
                # does when gathering scores from its folds.
                remaining = iter(fresh)
                ordered = np.asarray(
                    [
                        int(values[reused[i]]) if i in reused else int(next(remaining))
                        for i in range(len(splits))
                    ]
                )
                return float(score_mean(ordered)), ordered, len(reused)

        score, values = self._score_final_candidate(params, splits)
        return score, values, 0

    def _validate_optimization_settings(self):
        _validate_optimizer_control(
            self.local_search_top_k,
            self.local_search_steps,
            self.local_search_radius,
            self.diversity_threshold,
            self.diversity_stagnation_generations,
            self.diversity_mutation_boost,
            self.random_immigrants_fraction,
            self.sharing_radius,
            self.sharing_alpha,
            self.selection_pressure_min,
            self.selection_pressure_max,
            self.offspring_diversity_retries,
        )

    def _validate_final_selection_mode(self):
        self._validate_final_selection_top_k()
        if callable(self.refit) and self.final_selection:
            raise ValueError("Callable refit requires final_selection=False")
        if isinstance(self.final_selection, str):
            if self.final_selection != "reuse_folds":
                raise ValueError("final_selection must be False, True, or 'reuse_folds'")
            if self.error_score != "raise":
                raise ValueError("final_selection='reuse_folds' requires error_score='raise'")

        cv = self.final_selection_cv
        if not self.refit or not self.final_selection or cv is None:
            return
        if isinstance(cv, Integral):
            if not 2 <= int(cv) <= _num_samples(self.X_):
                raise ValueError("final_selection_cv must be between 2 and the number of samples")
        elif isinstance(cv, (str, bytes)) or (
            not callable(getattr(cv, "split", None)) and not isinstance(cv, Iterable)
        ):
            raise ValueError(
                "final_selection_cv must be an integer, a splitter, or a fold iterable"
            )
        elif type(cv) in (list, tuple):
            _cv_split_count(cv, "final_selection_cv")
            _validate_explicit_folds(cv, _num_samples(self.X_), parameter="final_selection_cv")

    def _select_final_candidate(self, *, allow_fold_reuse=True, original_best_index=None):
        indices = self._top_candidate_indices() if self.final_selection else None
        if indices is not None:
            original_best_index = int(indices[0])
        elif original_best_index is None:
            original_best_index = self._best_ranked_index()
        original_best_score = float(
            self.cv_results_[f"mean_test_{self.refit_metric}"][original_best_index]
        )
        original_best_params = self._candidate_params_from_index(original_best_index)

        self.final_selection_results_ = {
            "enabled": bool(self.final_selection),
            "top_k": 1,
            "cv": self.final_selection_cv,
            "original_best_index": original_best_index,
            "original_best_score": original_best_score,
            "original_best_params": original_best_params,
            "selected_index": original_best_index,
            "selected_score": original_best_score,
            "selected_params": original_best_params,
            "changed": False,
            "candidates": [],
            "time_seconds": 0.0,
        }

        if callable(self.refit):
            selected_index = self.refit(self.cv_results_)
            if isinstance(selected_index, (bool, np.bool_)) or not isinstance(
                selected_index, Integral
            ):
                raise TypeError("refit callable must return an integer index into cv_results_")
            if not 0 <= selected_index < len(self.cv_results_[f"rank_test_{self.refit_metric}"]):
                raise IndexError("refit callable returned an index outside cv_results_")
            selected_index = int(selected_index)
            selected_params = self._candidate_params_from_index(selected_index)
            self.final_selection_results_.update(
                selected_index=selected_index,
                selected_score=float(
                    self.cv_results_[f"mean_test_{self.refit_metric}"][selected_index]
                ),
                selected_params=selected_params,
                changed=selected_index != original_best_index,
            )
            return selected_index, None, selected_params

        if not self.final_selection:
            return original_best_index, original_best_score, original_best_params

        candidate_results = []
        selection = self.final_selection_results_
        selection.update(
            top_k=self.final_selection_top_k,
            selected_index=None,
            selected_score=None,
            selected_params=None,
            candidates=candidate_results,
        )
        started_at = time.monotonic()
        try:
            cv_splits = self._final_selection_splits()
            reused = (
                self._matching_search_folds(cv_splits)
                if self.final_selection == "reuse_folds" and allow_fold_reuse
                else {}
            )
            missing = [fold for position, fold in enumerate(cv_splits) if position not in reused]
            counts = self.fit_counts_
            for index in indices:
                params = self._candidate_params_from_index(index)
                previous_calls = counts["finalist_cv_calls"]
                previous_fits = counts["finalist_fold_fits"]
                score, cv_scores, reuse_count = self._score_final_folds(
                    index, params, cv_splits, reused, missing
                )
                counts["finalist_reused_folds"] += reuse_count
                candidate_results.append(
                    {
                        "index": int(index),
                        "original_score": float(
                            self.cv_results_[f"mean_test_{self.refit_metric}"][index]
                        ),
                        "score": score,
                        "cv_scores": cv_scores.tolist(),
                        "cv_calls": counts["finalist_cv_calls"] - previous_calls,
                        "fold_fits": counts["finalist_fold_fits"] - previous_fits,
                        "reused_folds": reuse_count,
                        "params": params,
                    }
                )
        finally:
            selection["time_seconds"] = time.monotonic() - started_at

        valid_results = [item for item in candidate_results if np.isfinite(item["score"])]
        if not valid_results:
            raise ValueError("All final-selection candidates produced non-finite scores")
        direction = -1 if self.criteria == "min" else 1
        selected = max(
            valid_results,
            key=lambda item: (
                direction * item["score"],
                -sum(item["params"]["features"]) if not hasattr(self, "space") else 0,
            ),
        )
        selected_index = selected["index"]
        selected_score = selected["score"]
        selected_params = selected["params"]

        selection.update(
            selected_index=selected_index,
            selected_score=selected_score,
            selected_params=selected_params,
            changed=selected_index != original_best_index,
        )

        return selected_index, selected_score, selected_params


class GASearchCV(_FinalSelectionMixin, GeneticEstimatorMixin, BaseSearchCV):
    """
    Evolutionary optimization over hyperparameters.

    GASearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "predict_log_proba" if they are implemented in the
    estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    Parameters
    ----------
    estimator : estimator object, default=None
        estimator object implementing 'fit'
        Template to clone for candidate evaluation and final refit. The supplied
        object is not fitted or modified; the fitted winner is ``best_estimator_``.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding at least one (train, test) split as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    param_grid : dict, default=None
        Grid with the parameters to tune, expects keys a valid name
        of hyperparameter based on the estimator selected and as values
        one of :class:`~sklearn_genetic.space.Integer` ,
        :class:`~sklearn_genetic.space.Categorical`
        :class:`~sklearn_genetic.space.Continuous` classes.
        At least two parameters are advised to be provided in order to successfully make
        an optimization routine.

    population_size : int, default=10
        Size of the initial population to sample generated individuals.

    evolution_config : :class:`~sklearn_genetic.config.EvolutionConfig`, default=None
        Optional grouped configuration for core genetic algorithm controls such
        as population size, generation count, crossover, mutation, tournament
        size, elitism, hall-of-fame size, criteria, and algorithm.

    population_config : :class:`~sklearn_genetic.config.PopulationConfig`, default=None
        Optional grouped configuration for initial population behavior,
        including ``initializer`` and ``warm_start_configs``.

    runtime_config : :class:`~sklearn_genetic.config.RuntimeConfig`, default=None
        Optional grouped configuration for parallelism, caching, train-score
        collection, error handling, and verbose output.

    optimization_config : :class:`~sklearn_genetic.config.OptimizationConfig`, default=None
        Optional grouped configuration for local refinement, diversity control,
        adaptive selection, fitness sharing, and robust final selection.

    population_initializer : {'smart', 'random'}, default='smart'
        Strategy used to generate the initial population. ``'smart'`` combines
        valid warm-start configurations, valid estimator defaults, Latin
        hypercube sampling for numeric dimensions, independently shuffled
        categorical strata across the full domain, and duplicate avoidance.
        ``'random'`` uses the previous random
        sampling behavior.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the search. When set, it seeds every
        stochastic step from a single place at ``fit`` time — population
        initialization (including the Latin hypercube sampler), mutation,
        crossover, and random immigrants — so repeated fits give identical
        results without manually seeding the global ``random`` / ``numpy`` RNGs.
        ``None`` keeps the default non-deterministic behaviour.

    generations : int, default=40
        Maximum number of generations. With caching enabled, evolution can
        stop earlier once every configuration in a finite space has been
        observed by the current run.

    crossover_probability : float or a Scheduler, default=0.8
        Probability of crossover operation between two individuals.

    mutation_probability : float or a Scheduler, default=0.1
        Probability of child mutation.

    tournament_size : int, default=3
        Number of individuals to perform tournament selection.

    elitism : bool, default=True
        If True takes the *tournament_size* best solution to the next generation.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:

        - a single string;
        - a callable that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a dictionary mapping metric names to scorer strings or callables.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Candidate evaluations in each
        generation are parallelized when possible; each candidate then runs
        cross-validation sequentially to avoid nested parallelism.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. ``0`` raises an error at fit time.

    parallel_backend : {'auto', 'population', 'cv'}, default='auto'
        Controls where ``n_jobs`` parallelism is applied during ``fit``.
        ``'auto'`` and ``'population'`` evaluate unique candidates in each
        generation in parallel when possible. ``'cv'`` keeps candidate
        evaluation serial and passes ``n_jobs`` to each candidate's
        cross-validation call.

    local_search : bool, default=False
        If ``True``, run a short local refinement phase around the current
        hall-of-fame individuals after the genetic search finishes.

    local_search_top_k : int, default=1
        Number of hall-of-fame individuals used as local-search seeds.

    local_search_steps : int, default=1
        Number of sequential refinement steps. Each step evaluates one
        neighbor per current elite and uses accepted improvements in the next
        step. Independent elites can be evaluated in parallel within a step.

    local_search_radius : float, default=0.1
        Fraction of the search range used to sample local numeric neighbors.
        For categorical parameters, a different category is sampled.

    diversity_control : bool, default=True
        If ``True``, monitor diversity and stagnation to boost mutation,
        replace duplicate candidates, and inject random immigrants.

    adaptive_selection : bool, default=False
        If ``True``, adapt tournament size from generation telemetry. Selection
        pressure is reduced when diversity is low or the search is stagnant,
        and slightly increased when the population is improving with enough
        diversity.

    selection_pressure_min : int, default=2
        Minimum tournament size used by adaptive selection.

    selection_pressure_max : int, default=None
        Maximum tournament size used by adaptive selection. If ``None``, the
        maximum is one larger than ``tournament_size``.

    offspring_diversity_retries : int, default=0
        Number of retries used when replacing duplicate or parent-matching
        offspring with new random candidates.

    diversity_threshold : float, default=0.25
        Diversity value below which diversity control can trigger.

    diversity_stagnation_generations : int, default=5
        Number of stagnant generations after which diversity control can
        inject random immigrants.

    diversity_mutation_boost : float, default=2.0
        Multiplicative boost applied to mutation probability when diversity
        control triggers. The boosted value is capped to DEAP's valid range.

    random_immigrants_fraction : float, default=0.1
        Fraction of offspring replaced by random immigrants when diversity
        control triggers.

    fitness_sharing : bool, default=False
        If ``True``, temporarily penalize candidates in crowded niches during
        selection. Raw cross-validation scores and ``cv_results_`` are not
        modified.

    sharing_radius : float, default=0.2
        Normalized distance below which two individuals are considered part of
        the same niche for fitness sharing.

    sharing_alpha : float, default=1.0
        Shape parameter that controls how quickly sharing pressure decreases
        with distance inside ``sharing_radius``.

    final_selection : bool or {"reuse_folds"}, default=False
        If ``True``, re-evaluate the top ``final_selection_top_k`` candidates
        after the GA finishes and select ``best_params_`` from those robust
        final scores before refitting.

        For deterministic estimators and scoring, set to ``"reuse_folds"`` with
        ``error_score="raise"`` to reuse identical training and validation splits
        from the same search. Resumed searches evaluate finalists afresh.

    final_selection_top_k : int, default=3
        Number of top candidates from the original GA ``cv_results_`` to
        re-evaluate during final selection.

    final_selection_cv : int, cross-validation splitter or iterable, default=None
        Cross-validation strategy used for final selection. If ``None``, the
        same CV splits used during the GA are reused. An explicit non-None
        value takes precedence over ``optimization_config.final_selection_cv``.

    verbose : bool, default=True
        If ``True``, shows the metrics on the optimization routine.

    keep_top_k : int, default=1
        Number of best solutions to keep in the hof object. If a callback stops the algorithm before k iterations,
        it will return only one set of parameters per iteration.

    criteria : {'max', 'min'} , default='max'
        ``max`` if a higher scoring metric is better, ``min`` otherwise.

    optimizer_backend : str or callable, default="deap"
        Keyword-only engine selector. Use ``"deap"`` (default), ``"scipy"``,
        ``"nevergrad"``, ``"cmaes"``, ``"pymoo"``, an installed optimizer entry
        point or a callable implementing ``sklearn_genetic.backends.Optimizer``.
        Install the corresponding extra for each alternative engine. The
        estimator owns CV, caching and refit. ``parallel_backend`` controls
        evaluation parallelism. DEAP-specific variation controls require DEAP.

    algorithm : str, default='eaMuPlusLambda'
        One of ``eaMuPlusLambda``, ``eaMuCommaLambda``, ``eaSimple``, or
        ``eaDifferentialEvolution``. Differential evolution uses deferred
        rand/1/bin trials with scale 0.8 and requires population_size >= 4.
        Each generation proposes one trial per parent and retains the better
        weighted fitness (the parent on ties). Numeric differences respect
        logarithmic spaces; categorical values come from donors, and feature
        masks use binary decoding followed by feature-count repair.
        crossover_probability controls trial coordinates; mutation_probability
        adds the existing mutation operator. Tournament selection, elitism,
        adaptive_selection and fitness_sharing apply only to the other three
        algorithms. Local refinement and diversity recovery remain available.

    refit : bool, str or callable, default=True
        Refit the selected candidate on the whole dataset. For multi-metric
        scoring, a scorer name guides the search and ranks candidates.
        Use ``False`` to skip finalist validation and the full-data fit.
        The selected search row and its score remain available.
        With ``final_selection=False``, a callable can select the candidate
        by returning an integer row index from ``cv_results_``. For multiple
        metrics and a callable refit, set ``objective_metric`` to guide the
        search and to choose the metric returned by ``score``.

    objective_metric : str or None, default=None
        Keyword-only name of the scorer that guides a multi-metric search
        with ``refit=False`` or callable ``refit``. With a named ``refit``,
        either omit this option or use the same scorer name.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be ``None`` to dispatch all
        jobs immediately, an integer number of total jobs to spawn, or a string
        expression as a function of ``n_jobs``, such as ``'2*n_jobs'``.

    error_score : 'raise' or numeric, default=np.nan
        Replacement score when estimator fitting fails on a CV fold. With a
        numeric value, a candidate that fails every fold emits
        ``FitFailedWarning`` and ranks last with ``NaN`` scores.
        Set to ``'raise'`` to propagate the original exception.

    return_train_score: bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    log_config : :class:`~sklearn_genetic.mlflow.MLflowConfig`, default = None
        Configuration to log metrics and models to mlflow, of None,
        no mlflow logging will be performed

    use_cache: bool, default=True
        If set to true it will avoid to re-evaluating solutions that have already seen,
        otherwise it will always evaluate the solutions to get the performance metrics

    Attributes
    ----------
    fit_counts_ : dict
        Search CV calls and fold fits, finalist CV calls and fresh or reused
        folds, and full-data refits in the most recent fit. On checkpoint resume
        the fresh work excludes evaluations restored from the checkpoint.

    search_time_ : float
        Elapsed seconds of the optimizer run in the most recent fit, including
        initial population setup, candidate evaluation, and callbacks. Available
        after the optimizer returns or raises.

    search_space_exhausted_ : bool
        Whether the search covered a finite search space at a generation
        boundary. Requires ``use_cache=True``. SciPy, Nevergrad, CatCMAwM and
        pymoo count completed generations of an exact resume and check
        coverage when finalist selection is off or ``refit=False``.

    logbook : :class:`DEAP.tools.Logbook`
        Compatibility view of per-candidate evaluation records. Shared-runtime
        engines construct the DEAP view when this attribute is accessed.
    history : dict
        Dictionary with one list per generation. It includes ``gen``,
        ``fitness``, ``fitness_std``, ``fitness_best``, ``fitness_max``, ``fitness_min``,
        population diversity fields, stagnation fields, optimizer-control
        telemetry, and local-refinement telemetry.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score
        on the left out data. Not available if ``refit=False``.
    best_params_ : dict
        Parameters from the selected row of ``cv_results_``.
    best_index_ : int
        Row in ``cv_results_`` for the selected candidate, including with
        ``refit=False``.
    best_score_ : float
        Mean objective score of the selected candidate. Uses the finalist
        score when finalist validation runs. With callable refit, read the
        objective score from the selected row of ``cv_results_``.
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.
    fit_stats_ : dict
        Counters collected during the last ``fit`` call. Includes evaluated
        candidates, unique candidates, cross-validation calls, cache hits,
        duplicate candidates, skipped invalid candidates, and population-level
        parallel/serial batch counts.
    """

    # Keep the search splitter's implicit requests. The search itself only
    # consumes routed groups directly when the caller opts in.
    __metadata_request__fit = {"groups": False}

    def __init__(
        self,
        estimator,
        cv=3,
        param_grid=None,
        scoring=None,
        population_size=50,
        generations=80,
        crossover_probability=0.8,
        mutation_probability=0.1,
        tournament_size=3,
        elitism=True,
        verbose=True,
        keep_top_k=1,
        criteria="max",
        algorithm="eaMuPlusLambda",
        refit=True,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        log_config=None,
        use_cache=True,
        warm_start_configs=None,
        evolution_config=None,
        population_config=None,
        runtime_config=None,
        optimization_config=None,
        parallel_backend="auto",
        population_initializer="smart",
        random_state=None,
        local_search=False,
        local_search_top_k=1,
        local_search_steps=1,
        local_search_radius=0.1,
        diversity_control=True,
        diversity_threshold=0.25,
        diversity_stagnation_generations=5,
        diversity_mutation_boost=2.0,
        random_immigrants_fraction=0.1,
        adaptive_selection=False,
        selection_pressure_min=2,
        selection_pressure_max=None,
        offspring_diversity_retries=0,
        fitness_sharing=False,
        sharing_radius=0.2,
        sharing_alpha=1.0,
        final_selection=False,
        final_selection_top_k=3,
        final_selection_cv=None,
        *,
        optimizer_backend="deap",
        objective_metric=None,
    ):
        legacy_warm_start_configs = warm_start_configs

        population_size = _resolve_config_value(
            evolution_config, "population_size", population_size
        )
        generations = _resolve_config_value(evolution_config, "generations", generations)
        crossover_probability = _resolve_config_value(
            evolution_config, "crossover_probability", crossover_probability
        )
        mutation_probability = _resolve_config_value(
            evolution_config, "mutation_probability", mutation_probability
        )
        tournament_size = _resolve_config_value(
            evolution_config, "tournament_size", tournament_size
        )
        elitism = _resolve_config_value(evolution_config, "elitism", elitism)
        keep_top_k = _resolve_config_value(evolution_config, "keep_top_k", keep_top_k)
        criteria = _resolve_config_value(evolution_config, "criteria", criteria)
        algorithm = _resolve_config_value(evolution_config, "algorithm", algorithm)

        population_initializer = _resolve_config_value(
            population_config, "initializer", population_initializer
        )
        warm_start_configs = _resolve_config_value(
            population_config, "warm_start_configs", warm_start_configs
        )

        n_jobs = _resolve_config_value(runtime_config, "n_jobs", n_jobs)
        pre_dispatch = _resolve_config_value(runtime_config, "pre_dispatch", pre_dispatch)
        error_score = _resolve_config_value(runtime_config, "error_score", error_score)
        return_train_score = _resolve_config_value(
            runtime_config, "return_train_score", return_train_score
        )
        use_cache = _resolve_config_value(runtime_config, "use_cache", use_cache)
        parallel_backend = _resolve_config_value(
            runtime_config, "parallel_backend", parallel_backend
        )
        verbose = _resolve_config_value(runtime_config, "verbose", verbose)

        local_search = _resolve_config_value(optimization_config, "local_search", local_search)
        local_search_top_k = _resolve_config_value(
            optimization_config, "local_search_top_k", local_search_top_k
        )
        local_search_steps = _resolve_config_value(
            optimization_config, "local_search_steps", local_search_steps
        )
        local_search_radius = _resolve_config_value(
            optimization_config, "local_search_radius", local_search_radius
        )
        diversity_control = _resolve_config_value(
            optimization_config, "diversity_control", diversity_control
        )
        diversity_threshold = _resolve_config_value(
            optimization_config, "diversity_threshold", diversity_threshold
        )
        diversity_stagnation_generations = _resolve_config_value(
            optimization_config,
            "diversity_stagnation_generations",
            diversity_stagnation_generations,
        )
        diversity_mutation_boost = _resolve_config_value(
            optimization_config, "diversity_mutation_boost", diversity_mutation_boost
        )
        random_immigrants_fraction = _resolve_config_value(
            optimization_config, "random_immigrants_fraction", random_immigrants_fraction
        )
        adaptive_selection = _resolve_config_value(
            optimization_config, "adaptive_selection", adaptive_selection
        )
        selection_pressure_min = _resolve_config_value(
            optimization_config, "selection_pressure_min", selection_pressure_min
        )
        selection_pressure_max = _resolve_config_value(
            optimization_config, "selection_pressure_max", selection_pressure_max
        )
        offspring_diversity_retries = _resolve_config_value(
            optimization_config, "offspring_diversity_retries", offspring_diversity_retries
        )
        fitness_sharing = _resolve_config_value(
            optimization_config, "fitness_sharing", fitness_sharing
        )
        sharing_radius = _resolve_config_value(
            optimization_config, "sharing_radius", sharing_radius
        )
        sharing_alpha = _resolve_config_value(optimization_config, "sharing_alpha", sharing_alpha)
        final_selection = _resolve_config_value(
            optimization_config, "final_selection", final_selection
        )
        final_selection_top_k = _resolve_config_value(
            optimization_config, "final_selection_top_k", final_selection_top_k
        )
        self._explicit_finalist_cv = final_selection_cv
        if final_selection_cv is None:
            final_selection_cv = _resolve_config_value(
                optimization_config, "final_selection_cv", final_selection_cv
            )

        self.evolution_config = evolution_config
        self.population_config = population_config
        self.runtime_config = runtime_config
        self.optimization_config = optimization_config
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self._configure_adapters()
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.verbose = verbose
        self.keep_top_k = keep_top_k
        self.criteria = criteria
        self.param_grid = param_grid
        self.optimizer_backend = optimizer_backend
        self.algorithm = algorithm
        self.refit = refit
        self.objective_metric = objective_metric
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.log_config = log_config
        self.use_cache = use_cache
        self.fitness_cache = {}
        self.warm_start_configs = legacy_warm_start_configs
        self._warm_start_configs = warm_start_configs
        self.parallel_backend = parallel_backend
        self.population_initializer = population_initializer
        self.random_state = random_state
        self.local_search = local_search
        self.local_search_top_k = local_search_top_k
        self.local_search_steps = local_search_steps
        self.local_search_radius = local_search_radius
        self.diversity_control = diversity_control
        self.diversity_threshold = diversity_threshold
        self.diversity_stagnation_generations = diversity_stagnation_generations
        self.diversity_mutation_boost = diversity_mutation_boost
        self.random_immigrants_fraction = random_immigrants_fraction
        self.adaptive_selection = adaptive_selection
        self.selection_pressure_min = selection_pressure_min
        self.selection_pressure_max = selection_pressure_max
        self.offspring_diversity_retries = offspring_diversity_retries
        self.fitness_sharing = fitness_sharing
        self.sharing_radius = sharing_radius
        self.sharing_alpha = sharing_alpha
        self.final_selection = final_selection
        self.final_selection_top_k = final_selection_top_k
        self.final_selection_cv = final_selection_cv

        _validate_parallel_backend(self.parallel_backend)
        _validate_error_score(self.error_score)
        _validate_population_initializer(self.population_initializer)
        self._validate_final_selection_top_k()
        self._validate_optimization_settings()

        # Check that the estimator is compatible with scikit-learn
        if not (
            _is_classifier(self.estimator)
            or _is_regressor(self.estimator)
            or _is_outlier_detector(self.estimator)
        ):
            raise ValueError(
                f"{self.estimator} is not a valid Sklearn classifier, regressor, or outlier detector"
            )

        self.criteria_sign = _get_criteria_sign(self.criteria)

        # Saves the param_grid and computes some extra properties in the same object
        self.space = Space(param_grid)

        super(GASearchCV, self).__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _register(self):
        from ._deap_backend import register_parameters

        self.criteria_sign = _get_criteria_sign(self.criteria)
        register_parameters(self)

    def _initialize_population(self):
        """
        Initialize the population, using warm-start configurations if provided.
        """
        from ._deap_backend import initialize_population

        return initialize_population(self, initialize_search_population)

    def select(self, population, k):
        from ._deap_backend import select_tournament

        return select_tournament(self, population, k)

    _repair_value = staticmethod(Space._repair_value)

    def _repair_individual(self, individual):
        if not hasattr(self, "space"):
            return individual

        for index, dimension in enumerate(self.space.param_grid.values()):
            individual[index] = self._repair_value(dimension, individual[index])

        return individual

    def mate(self, individual_1, individual_2):
        offspring_1, offspring_2 = self.toolbox.mate_raw(individual_1, individual_2)
        self._repair_individual(offspring_1)
        self._repair_individual(offspring_2)
        return offspring_1, offspring_2

    def mutate(self, individual):
        """
        This function is responsible for change a randomly selected parameter from an individual
        Parameters
        ----------
        individual: Individual object
            The individual (set of hyperparameters) that is being generated

        Returns
        -------
            Mutated individual
        """

        mutable = mutable_dimensions(self.space.param_grid.values())
        if mutable:
            index, dimension = random.choice(mutable)
            individual[index] = dimension.sample()
        self._repair_individual(individual)

        return [individual]

    def _same_individual(self, left, right):
        return self._individual_key(left) == self._individual_key(right)

    def _individual_key(self, individual):
        return self.space._candidate_key(individual)

    def evaluate_population(self, individuals):
        for individual in individuals:
            self._repair_individual(individual)
        if _uses_prepared_integer_evaluation(self):
            return _evaluate_population_batch(
                self,
                individuals,
                "current_generation_params",
                evaluate=self._evaluate_prepared_individual,
            )
        return _evaluate_population_batch(self, individuals, "current_generation_params")

    def _evaluate_individual(self, individual, n_jobs=None):
        self._repair_individual(individual)
        return self._evaluate_prepared_individual(individual, n_jobs=n_jobs)

    def _evaluate_prepared_individual(self, individual, n_jobs=None):
        current_generation_params = {
            name: individual[index] for index, name in enumerate(self.space.param_grid.keys())
        }

        local_estimator = clone(self.estimator)
        local_estimator.set_params(**_clone_mutable_candidate_params(current_generation_params))
        if self.scorer_ is _default_outlier_scorer:
            _outlier_response(local_estimator)

        cv_results = self._cross_validate_candidate(
            local_estimator,
            self.X_,
            self._cv_splits,
            n_jobs=n_jobs,
            return_train_score=self.return_train_score,
        )

        cv_scores = cv_results[f"test_{self.refit_metric}"]
        score = score_mean(cv_scores)

        # Uses the log config to save in remote log server (e.g MLflow)
        if self.log_config is not None and cv_results["fit_time"] is not None:
            self.log_config.create_run(
                parameters=current_generation_params,
                score=score,
                estimator=local_estimator,
            )

        conflicts = getattr(self, "_candidate_parameter_conflicts", None)
        if conflicts is None:
            conflicts = parameter_record_conflicts(
                current_generation_params, self.metrics_list, self.return_train_score
            )
        if conflicts:
            current_generation_params = preserve_candidate_parameters(
                current_generation_params, conflicts
            )

        # These values are used to compute cv_results_ property
        current_generation_params["score"] = score
        current_generation_params["cv_scores"] = cv_scores
        current_generation_params["fit_time"] = cv_results["fit_time"]
        current_generation_params["score_time"] = cv_results["score_time"]

        for metric in self.metrics_list:
            current_generation_params[f"test_{metric}"] = cv_results[f"test_{metric}"]

            if self.return_train_score:
                current_generation_params[f"train_{metric}"] = cv_results[f"train_{metric}"]

        fitness_result = (_evolution_score(score, self.criteria_sign),)

        return fitness_result, current_generation_params, True, False

    def evaluate(self, individual):
        """
        Compute the cross-validation scores and record the logbook and mlflow (if specified)
        Parameters
        ----------
        individual: Individual object
            The individual (set of hyperparameters) that is being evaluated
        Returns
        -------
            The fitness value of the estimator candidate, corresponding to the cv-score

        """

        # Convert hyperparameters to a tuple to use as a key in the cache
        self._repair_individual(individual)
        individual_key = self._individual_key(individual) if self.use_cache else None

        # Check if the individual has already been evaluated
        if self.use_cache and individual_key in self.fitness_cache:
            # Retrieve cached result
            cached_result = self.fitness_cache[individual_key]
            # Ensure the logbook is updated even if the individual is cached
            record_candidate(self, cached_result["current_generation_params"])
            _record_fit_stats(self, evaluated=1, cache_hits=1)
            return cached_result["fitness"]

        candidate_n_jobs = self.n_jobs if self.parallel_backend == "cv" else 1
        (
            fitness_result,
            current_generation_params,
            used_cv,
            skipped_invalid,
        ) = self._evaluate_individual(
            individual,
            n_jobs=candidate_n_jobs,
        )
        current_generation_params = record_candidate(
            self,
            current_generation_params,
        )

        if self.use_cache:
            # Store the fitness result and the current generation parameters in the cache
            self.fitness_cache[individual_key] = {
                "fitness": fitness_result,
                "current_generation_params": current_generation_params,
            }

        _record_fit_stats(
            self,
            evaluated=1,
            unique=1,
            cv_calls=int(used_cv),
            skipped=int(skipped_invalid),
        )

        return fitness_result

    def fit(self, X, y=None, callbacks=None, groups=None, **fit_params):
        """
        Main method of GASearchCV, starts the optimization
        procedure with the hyperparameters of the given estimator

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
            Pass the target for supervised estimators or pipeline steps
            that need it. Omit y for unlabeled outlier detection.
        callbacks: list or callable
            One or a list of the callbacks methods available in
            :class:`~sklearn_genetic.callbacks`.
            The callback is evaluated after fitting the estimators from the generation 1.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test sets. Only used in conjunction with a "Group" cv
            instance such as :class:`~sklearn.model_selection.GroupKFold`.
        **fit_params : dict of str -> object
            Parameters for candidate and finalist cross-validation and final
            refit, such as ``sample_weight``. With metadata routing enabled,
            scikit-learn routes them to the estimator and scorers according
            to their requests. Sample-aligned parameters are sliced per CV
            fold. The final refit receives the estimator's fit parameters.
        """

        self._clear_fit_details()
        _check_required_target(self.estimator, y)
        self.X_ = X
        self.y_ = y
        self.groups_ = groups
        self._fit_params = fit_params
        self._warm_start_configs = _resolve_config_value(
            self.population_config, "warm_start_configs", self.warm_start_configs
        )
        _check_fit_params_supported(self.estimator, fit_params)

        # Make sure the callbacks are valid
        self.callbacks = check_callback(callbacks)
        optimizer = self._validate_backend()

        checkpoint, context = self._prepare_checkpoint(optimizer)
        legacy_checkpoint = checkpoint if optimizer is None else None
        runtime = _restore_legacy_checkpoint(self, legacy_checkpoint)
        restored_adapter_state = runtime.get("adapter_state")
        if optimizer is None:
            _validate_search_counts(self, "DEAP")
        self._n_iterations = int(self.generations) + 1

        _validate_parallel_backend(self.parallel_backend, n_jobs=self.n_jobs)
        _validate_error_score(self.error_score)
        _validate_population_initializer(self.population_initializer)
        # ``set_params(param_grid=...)`` updates the public parameter but not
        # the Space built at construction. Rebuild after any checkpoint load,
        # which may restore its own param_grid for a resumed search.
        self.space = Space(self.param_grid)
        if (
            checkpoint is not None
            and not checkpoint.get("typed_categorical_identity")
            and self.space._typed_categories
        ):
            raise ValueError(
                "This checkpoint may contain merged categorical scores for "
                "equal values of different types. Use a new checkpoint path."
            )
        # Seed after the checkpoint (if any) is loaded: ``self.__dict__.update()``
        # above may have just replaced ``self.random_state`` with the value from
        # a resumed run's checkpoint, and seeding before that point would use the
        # pre-resume value instead (see #299).
        _seed_global_rngs(self.random_state)

        self._validate_optimization_settings()
        self._validate_final_selection_mode()
        self._configure_adapters()
        if restored_adapter_state is not None:
            # Restore the crossover/mutation adapter step counters so the
            # schedule continues from where the previous run left off
            # instead of restarting from step 0 (see adapter checkpoint).
            self.crossover_adapter.load_state_dict(restored_adapter_state["crossover"])
            self.mutation_adapter.load_state_dict(restored_adapter_state["mutation"])
        elif legacy_checkpoint is None:
            self.fitness_cache = {}
            _reset_adapters(self)

        # Preserve cumulative counters across a resume instead of zeroing them,
        # so e.g. ``cache_hits``/``evaluated_candidates`` reflect the whole run.
        self.fit_stats_ = runtime.get("fit_stats_") or _create_fit_stats()
        self._resume_generation_log = (
            legacy_checkpoint.get("logbook") if legacy_checkpoint else None
        )

        self._set_search_scoring(checkpoint)
        conflicts = parameter_record_conflicts(
            self.space.parameters, self.metrics_list, self.return_train_score
        )
        self._candidate_parameter_conflicts = conflicts
        if checkpoint is not None and conflicts - {"index"}:
            records = checkpoint.get("candidate_records")
            if records is None:
                book = runtime.get("candidate_logbook")
                records = book.chapters["parameters"] if book is not None else ()
            if any(candidate_parameters(row) is row for row in records):
                raise ValueError(
                    "Checkpoint did not preserve candidate parameters that overlap CV fields. "
                    "Use a new checkpoint path."
                )
        if legacy_checkpoint is not None and not legacy_checkpoint.get("stable_nan_categories"):
            self.fitness_cache = self.space._rekey_nan_cache(self.fitness_cache)

        self._set_search_splits(checkpoint, optimizer, runtime)

        _restore_search_rngs(runtime.get("rng_state"))
        self.criteria_sign = _get_criteria_sign(self.criteria)
        log, n_gen = self._run_optimizer(optimizer, runtime, checkpoint, context)

        # Update the _n_iterations value as the algorithm could stop earlier due a callback
        self._n_iterations = n_gen

        self.cv_results_ = create_gasearch_cv_results_(
            records=candidate_records(self),
            space=self.space,
            return_train_score=self.return_train_score,
            metrics=self.metrics_list,
            greater_is_better=self.criteria == "max",
            unique_candidates=self.fit_stats_.get("unique_candidates"),
        )

        self._set_search_history(log)

        # Imitate the logic of scikit-learn refit parameter
        if self.refit:
            self.best_index_, best_score, self.best_params_ = self._select_final_candidate(
                allow_fold_reuse=legacy_checkpoint is None
                or (context is not None and self._copy_cv_indices)
            )
            if best_score is not None:
                self.best_score_ = best_score

            best_estimator = clone(self.estimator).set_params(
                **_clone_mutable_candidate_params(self.best_params_)
            )

            refit_params = self._refit_metadata(best_estimator)
            targets = self._target_args(self.y_)
            refit_start_time = time.monotonic()
            best_estimator.fit(self.X_, *targets, **refit_params)
            self.refit_time_ = time.monotonic() - refit_start_time
            self.fit_counts_["refit_fits"] = 1

            self.best_estimator_ = self.estimator_ = best_estimator
        else:
            self.best_index_ = self._best_ranked_index()
            self.best_params_ = self._candidate_params_from_index(self.best_index_)
            self.best_score_ = float(
                self.cv_results_[f"mean_test_{self.refit_metric}"][self.best_index_]
            )

        hof_candidates = list(self._hof)
        if self.refit:
            # Put the selected winner first without duplicating an existing elite.
            best = [self.best_params_[name] for name in self.space.parameters]
            hof_candidates = [best] + [
                candidate
                for candidate in hof_candidates
                if not self._same_individual(best, candidate)
            ]
        self.hof = {
            k: dict(zip(self.space.parameters, candidate))
            for k, candidate in enumerate(hof_candidates[: self.keep_top_k])
        }

        return self

    def get_metadata_routing(self):
        """Expose explicit fit requests alongside scikit-learn's CV routing."""
        request = self._get_metadata_request()
        router = super().get_metadata_routing()
        if isinstance(request.fit.requests.get("groups"), str):
            # fit passes the aliased groups to both splitters. A concurrent
            # request from cv would also claim the outer splitter's groups.
            router.add(splitter=None, method_mapping=MethodMapping())
        return router.add_self_request(request)

    def score(self, X, y=None, **params):
        """Score the refitted estimator using the selected metric."""
        targets = self._target_args(y)
        callable_refit = callable(self.refit) and getattr(self, "multimetric_", False)
        if not self.refit or (targets and not callable_refit):
            return super().score(X, y, **params)

        check_is_fitted(self, "best_estimator_")
        scorer = self.scorer_[self.refit_metric] if self.multimetric_ else self.scorer_
        if params:
            _raise_for_params(params, self, "score")
            router = MetadataRouter(owner=type(self).__name__).add(
                scorer=scorer,
                method_mapping=MethodMapping().add(caller="score", callee="score"),
            )
            params = process_routing(router, "score", **params).scorer.score
        return scorer(self.best_estimator_, X, *targets, **params)


class GAFeatureSelectionCV(
    _FinalSelectionMixin, GeneticEstimatorMixin, MetaEstimatorMixin, SelectorMixin, BaseEstimator
):
    """
    Evolutionary optimization for feature selection.

    GAFeatureSelectionCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "predict_log_proba" if they are implemented in the
    estimator used.
    The features (variables) used by the estimator are found by optimizing
    the cv-scores and by minimizing the number of features

    Parameters
    ----------
    estimator : estimator object, default=None
        estimator object implementing 'fit'
        Template to clone for candidate evaluation and final refit. The supplied
        object is not fitted or modified; the fitted winner is ``best_estimator_``.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding at least one (train, test) split as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    population_size : int, default=10
        Size of the initial population to sample generated individuals.

    evolution_config : :class:`~sklearn_genetic.config.EvolutionConfig`, default=None
        Optional grouped configuration for core genetic algorithm controls such
        as population size, generation count, crossover, mutation, tournament
        size, elitism, hall-of-fame size, criteria, and algorithm.

    population_config : :class:`~sklearn_genetic.config.PopulationConfig`, default=None
        Optional grouped configuration for the initial feature-mask population.
        Pass ``warm_start_masks`` in input column order to evaluate known subsets
        before newly generated masks.

    runtime_config : :class:`~sklearn_genetic.config.RuntimeConfig`, default=None
        Optional grouped configuration for parallelism, caching, train-score
        collection, error handling, and verbose output.

    optimization_config : :class:`~sklearn_genetic.config.OptimizationConfig`, default=None
        Optional grouped configuration for local refinement, diversity control,
        adaptive selection, fitness sharing, and finalist selection.

    population_initializer : {'smart', 'random', 'spread'}, default='smart'
        Strategy used to generate the initial population. ``'smart'`` creates
        duplicate-aware feature masks with a range of selected-feature counts.
        ``'random'`` uses the previous weighted random feature-mask sampling.
        ``'spread'`` orders the smart masks so that early evaluations cover
        a wider range of counts, starting near the midpoint.

    local_search : bool, default=False
        If ``True``, run a short local refinement phase around the current
        hall-of-fame feature masks after the genetic search finishes.

    local_search_top_k : int, default=1
        Number of hall-of-fame feature masks used as local-search seeds.

    local_search_steps : int, default=1
        Number of sequential refinement steps. Each step evaluates one
        neighbor per current elite mask and uses accepted improvements in the
        next step. Independent masks can be evaluated in parallel within a step.

    local_search_radius : float, default=0.1
        Fraction of features to flip when sampling a local neighbor.

    diversity_control : bool, default=True
        If ``True``, monitor diversity and stagnation to boost mutation,
        replace duplicate candidates, and inject random immigrants.

    diversity_threshold : float, default=0.25
        Diversity value below which diversity control can trigger.

    diversity_stagnation_generations : int, default=5
        Number of stagnant generations after which diversity control can
        inject random immigrants.

    diversity_mutation_boost : float, default=2.0
        Multiplicative boost applied to mutation probability when diversity
        control triggers. The boosted value is capped to DEAP's valid range.

    random_immigrants_fraction : float, default=0.1
        Fraction of offspring replaced by random immigrants when diversity
        control triggers.

    adaptive_selection : bool, default=False
        If ``True``, adapt tournament size from generation telemetry. Selection
        pressure is reduced when diversity is low or the search is stagnant,
        and slightly increased when the population is improving with enough
        diversity.

    selection_pressure_min : int, default=2
        Minimum tournament size used by adaptive selection.

    selection_pressure_max : int, default=None
        Maximum tournament size used by adaptive selection. If ``None``, the
        maximum is one larger than ``tournament_size``.

    offspring_diversity_retries : int, default=0
        Number of retries used when replacing duplicate or parent-matching
        offspring with new random feature masks.

    fitness_sharing : bool, default=False
        If ``True``, temporarily penalize candidates in crowded niches during
        selection. Raw cross-validation scores and ``cv_results_`` are not
        modified.

    sharing_radius : float, default=0.2
        Normalized distance below which two individuals are considered part of
        the same niche for fitness sharing.

    sharing_alpha : float, default=1.0
        Shape parameter that controls how quickly sharing pressure decreases
        with distance inside ``sharing_radius``.

    generations : int, default=40
        Maximum number of generations. With caching enabled, evolution can
        stop earlier once every configuration in a finite space has been
        observed by the current run.

    crossover_probability : float or a Scheduler, default=0.8
        Probability of crossover operation between two individuals.

    mutation_probability : float or a Scheduler, default=0.1
        Probability of child mutation.

    tournament_size : int, default=3
        Number of individuals to perform tournament selection.

    elitism : bool, default=True
        If True takes the *tournament_size* best solution to the next generation.

    final_selection : bool or {"reuse_folds"}, default=False
        With ``refit=True``, re-evaluate the best distinct feature masks after
        evolution and choose the refitted winner using these final scores.
        Equal final scores prefer fewer features. Search scores in
        ``cv_results_`` and ``history`` remain unchanged; final scores are
        available in ``final_selection_results_`` and ``best_score_``.

        For deterministic estimators and scoring, set to ``"reuse_folds"`` with
        ``error_score="raise"`` to reuse identical training and validation splits
        from the same search. Resumed searches evaluate finalists afresh.

    final_selection_top_k : int, default=3
        Positive number of distinct masks to re-evaluate. This bounds extra
        cross-validation calls independently of ``keep_top_k``. If multiple
        finalist slots share the cutoff search score, include both a sparse
        and a dense mask from that tie. Better search scores retain priority;
        equal final scores still prefer fewer features.

    final_selection_cv : int, cross-validation splitter or iterable, default=None
        Validation used to choose among finalists. None reuses the search
        folds. Supply the original full-training folds when searching with
        :func:`subsample_cv`. An explicit non-None value takes precedence
        over ``optimization_config.final_selection_cv``.

    max_features : int, default=None
        The upper bound number of features to be selected. Must be None or an
        integer between 1 and the number of input features (validated at fit).

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:

        - a single string;
        - a callable that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a dictionary mapping metric names to scorer strings or callables.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Candidate evaluations in each
        generation are parallelized when possible; each candidate then runs
        cross-validation sequentially to avoid nested parallelism.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. ``0`` raises an error at fit time.

    verbose : bool, default=True
        If ``True``, shows the metrics on the optimization routine.

    keep_top_k : int, default=1
        Number of best solutions to keep in the hof object. If a callback stops the algorithm before k iterations,
        it will return only one set of parameters per iteration.

    criteria : {'max', 'min'} , default='max'
        ``max`` if a higher scoring metric is better, ``min`` otherwise.

    optimizer_backend : str or callable, default="deap"
        Keyword-only engine selector. Use ``"deap"`` (default), ``"scipy"``,
        ``"nevergrad"``, ``"cmaes"``, ``"pymoo"``, an installed optimizer entry
        point or a callable implementing ``sklearn_genetic.backends.Optimizer``.
        Install the corresponding extra for each alternative engine. The
        estimator owns CV, caching and refit. ``parallel_backend`` controls
        evaluation parallelism. DEAP-specific variation controls require DEAP.

    algorithm : str, default='eaMuPlusLambda'
        One of ``eaMuPlusLambda``, ``eaMuCommaLambda``, ``eaSimple``, or
        ``eaDifferentialEvolution``. Differential evolution uses deferred
        rand/1/bin trials with scale 0.8 and requires population_size >= 4.
        Each generation proposes one trial per parent and retains the better
        weighted fitness (the parent on ties). Numeric differences respect
        logarithmic spaces; categorical values come from donors, and feature
        masks use binary decoding followed by feature-count repair.
        crossover_probability controls trial coordinates; mutation_probability
        adds the existing mutation operator. Tournament selection, elitism,
        adaptive_selection and fitness_sharing apply only to the other three
        algorithms. Local refinement and diversity recovery remain available.

    refit : bool, str or callable, default=True
        Refit the selected candidate on the whole dataset. For multi-metric
        scoring, a scorer name guides the search and ranks candidates.
        Use ``False`` to skip finalist validation and the full-data fit.
        The selected search mask and its score remain available.
        With ``final_selection=False``, a callable can select the candidate
        by returning an integer row index from ``cv_results_``. For multiple
        metrics and a callable refit, set ``objective_metric`` to guide the
        search and to choose the metric returned by ``score``.

    objective_metric : str or None, default=None
        Keyword-only name of the scorer that guides a multi-metric search
        with ``refit=False`` or callable ``refit``. With a named ``refit``,
        either omit this option or use the same scorer name.

    preserve_dataframe : bool, default=False
        Pass selected pandas DataFrame columns, including their names and
        dtypes, to the wrapped estimator during validation, refit and
        prediction. This setting requires a DataFrame for these operations.
        Use ``set_output`` to control the container returned by ``transform``.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be ``None`` to dispatch all
        jobs immediately, an integer number of total jobs to spawn, or a string
        expression as a function of ``n_jobs``, such as ``'2*n_jobs'``.

    error_score : 'raise' or numeric, default=np.nan
        Replacement score when estimator fitting fails on a CV fold. With a
        numeric value, a mask that fails every fold emits ``FitFailedWarning``
        and ranks last with ``NaN`` scores.
        Set to ``'raise'`` to propagate the original exception.

    return_train_score: bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    log_config : :class:`~sklearn_genetic.mlflow.MLflowConfig`, default = None
        Configuration to log metrics and models to mlflow, of None,
        no mlflow logging will be performed

    use_cache: bool, default=True
        If set to true it will avoid to re-evaluating solutions that have already seen,
        otherwise it will always evaluate the solutions to get the performance metrics

    Attributes
    ----------
    fit_counts_ : dict
        Search CV calls and fold fits, finalist CV calls and fresh or reused
        folds, and full-data refits in the most recent fit. On checkpoint resume
        the fresh work excludes evaluations restored from the checkpoint.

    search_time_ : float
        Elapsed seconds of the optimizer run in the most recent fit, including
        initial population setup, candidate evaluation, and callbacks. Available
        after the optimizer returns or raises.

    search_space_exhausted_ : bool
        Whether the search covered a finite search space at a generation
        boundary. Requires ``use_cache=True``. SciPy, Nevergrad, CatCMAwM and
        pymoo count completed generations of an exact resume and check
        coverage when finalist selection is off or ``refit=False``.

    logbook : :class:`DEAP.tools.Logbook`
        Compatibility view of per-candidate evaluation records. Shared-runtime
        engines construct the DEAP view when this attribute is accessed.
    history : dict
        Dictionary with one list per generation. It includes ``gen``,
        ``fitness``, ``fitness_std``, ``fitness_best``, ``fitness_max``, ``fitness_min``,
        population diversity fields, stagnation fields, optimizer-control
        telemetry, and local-refinement telemetry.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
    best_estimator_ : estimator
        Estimator fitted on the training data with the selected features.
        Available when refitting is enabled.
    best_features_ : ndarray of bool
        Selected feature mask in the original column order.
    support_ : ndarray of bool
        The same mask of selected features.
    n_features_ : int
        Number of selected features in ``support_``, including after a search
        with ``refit=False``.
    best_score_ : float
        Mean search CV score for the selected mask, or its finalist CV score
        when finalist validation runs. Callable refit exposes the selected
        row instead of a separate best score.
    best_index_ : int
        Row in ``cv_results_`` corresponding to the selected mask. With
        standard refit or ``refit=False``, prefer the row that supplied the
        archived search score, when available.
    final_selection_results_ : dict
        Candidate masks, original scores, final fold scores, selected winner
        and elapsed validation time. Each finalist includes actual CV calls,
        fresh fold fits and reused scores. Callable refit records the selected row.
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    n_features_in_ : int
        Number of input features seen during fit, before selection.
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.
    fit_stats_ : dict
        Counters collected during the last ``fit`` call. Includes evaluated
        candidates, unique candidates, cross-validation calls, cache hits,
        duplicate candidates, skipped invalid candidates, and population-level
        parallel/serial batch counts.
    """

    # Older fitted instances do not store this constructor parameter.
    preserve_dataframe = False

    def __init__(
        self,
        estimator,
        cv=3,
        scoring=None,
        population_size=50,
        generations=80,
        crossover_probability=0.8,
        mutation_probability=0.1,
        tournament_size=3,
        elitism=True,
        max_features=None,
        verbose=True,
        keep_top_k=1,
        criteria="max",
        algorithm="eaMuPlusLambda",
        refit=True,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        log_config=None,
        use_cache=True,
        evolution_config=None,
        population_config=None,
        runtime_config=None,
        optimization_config=None,
        parallel_backend="auto",
        population_initializer="smart",
        random_state=None,
        local_search=False,
        local_search_top_k=1,
        local_search_steps=1,
        local_search_radius=0.1,
        diversity_control=True,
        diversity_threshold=0.25,
        diversity_stagnation_generations=5,
        diversity_mutation_boost=2.0,
        random_immigrants_fraction=0.1,
        adaptive_selection=False,
        selection_pressure_min=2,
        selection_pressure_max=None,
        offspring_diversity_retries=0,
        fitness_sharing=False,
        sharing_radius=0.2,
        sharing_alpha=1.0,
        final_selection=False,
        final_selection_top_k=3,
        final_selection_cv=None,
        *,
        optimizer_backend="deap",
        objective_metric=None,
        preserve_dataframe=False,
    ):
        population_size = _resolve_config_value(
            evolution_config, "population_size", population_size
        )
        generations = _resolve_config_value(evolution_config, "generations", generations)
        crossover_probability = _resolve_config_value(
            evolution_config, "crossover_probability", crossover_probability
        )
        mutation_probability = _resolve_config_value(
            evolution_config, "mutation_probability", mutation_probability
        )
        tournament_size = _resolve_config_value(
            evolution_config, "tournament_size", tournament_size
        )
        elitism = _resolve_config_value(evolution_config, "elitism", elitism)
        keep_top_k = _resolve_config_value(evolution_config, "keep_top_k", keep_top_k)
        criteria = _resolve_config_value(evolution_config, "criteria", criteria)
        algorithm = _resolve_config_value(evolution_config, "algorithm", algorithm)

        population_initializer = _resolve_config_value(
            population_config, "initializer", population_initializer
        )

        n_jobs = _resolve_config_value(runtime_config, "n_jobs", n_jobs)
        pre_dispatch = _resolve_config_value(runtime_config, "pre_dispatch", pre_dispatch)
        error_score = _resolve_config_value(runtime_config, "error_score", error_score)
        return_train_score = _resolve_config_value(
            runtime_config, "return_train_score", return_train_score
        )
        use_cache = _resolve_config_value(runtime_config, "use_cache", use_cache)
        parallel_backend = _resolve_config_value(
            runtime_config, "parallel_backend", parallel_backend
        )
        verbose = _resolve_config_value(runtime_config, "verbose", verbose)

        local_search = _resolve_config_value(optimization_config, "local_search", local_search)
        local_search_top_k = _resolve_config_value(
            optimization_config, "local_search_top_k", local_search_top_k
        )
        local_search_steps = _resolve_config_value(
            optimization_config, "local_search_steps", local_search_steps
        )
        local_search_radius = _resolve_config_value(
            optimization_config, "local_search_radius", local_search_radius
        )
        diversity_control = _resolve_config_value(
            optimization_config, "diversity_control", diversity_control
        )
        diversity_threshold = _resolve_config_value(
            optimization_config, "diversity_threshold", diversity_threshold
        )
        diversity_stagnation_generations = _resolve_config_value(
            optimization_config,
            "diversity_stagnation_generations",
            diversity_stagnation_generations,
        )
        diversity_mutation_boost = _resolve_config_value(
            optimization_config, "diversity_mutation_boost", diversity_mutation_boost
        )
        random_immigrants_fraction = _resolve_config_value(
            optimization_config, "random_immigrants_fraction", random_immigrants_fraction
        )
        adaptive_selection = _resolve_config_value(
            optimization_config, "adaptive_selection", adaptive_selection
        )
        selection_pressure_min = _resolve_config_value(
            optimization_config, "selection_pressure_min", selection_pressure_min
        )
        selection_pressure_max = _resolve_config_value(
            optimization_config, "selection_pressure_max", selection_pressure_max
        )
        offspring_diversity_retries = _resolve_config_value(
            optimization_config, "offspring_diversity_retries", offspring_diversity_retries
        )
        fitness_sharing = _resolve_config_value(
            optimization_config, "fitness_sharing", fitness_sharing
        )
        sharing_radius = _resolve_config_value(
            optimization_config, "sharing_radius", sharing_radius
        )
        sharing_alpha = _resolve_config_value(optimization_config, "sharing_alpha", sharing_alpha)
        final_selection = _resolve_config_value(
            optimization_config, "final_selection", final_selection
        )
        final_selection_top_k = _resolve_config_value(
            optimization_config, "final_selection_top_k", final_selection_top_k
        )
        self._explicit_finalist_cv = final_selection_cv
        if final_selection_cv is None:
            final_selection_cv = _resolve_config_value(
                optimization_config, "final_selection_cv", final_selection_cv
            )

        self.evolution_config = evolution_config
        self.population_config = population_config
        self.runtime_config = runtime_config
        self.optimization_config = optimization_config
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self._configure_adapters()
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.max_features = max_features
        self.preserve_dataframe = preserve_dataframe
        self.verbose = verbose
        self.keep_top_k = keep_top_k
        self.criteria = criteria
        self.optimizer_backend = optimizer_backend
        self.algorithm = algorithm
        self.refit = refit
        self.objective_metric = objective_metric
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.log_config = log_config
        self.use_cache = use_cache
        self.fitness_cache = {}
        self.parallel_backend = parallel_backend
        self.population_initializer = population_initializer
        self.random_state = random_state
        self.local_search = local_search
        self.local_search_top_k = local_search_top_k
        self.local_search_steps = local_search_steps
        self.local_search_radius = local_search_radius
        self.diversity_control = diversity_control
        self.diversity_threshold = diversity_threshold
        self.diversity_stagnation_generations = diversity_stagnation_generations
        self.diversity_mutation_boost = diversity_mutation_boost
        self.random_immigrants_fraction = random_immigrants_fraction
        self.adaptive_selection = adaptive_selection
        self.selection_pressure_min = selection_pressure_min
        self.selection_pressure_max = selection_pressure_max
        self.offspring_diversity_retries = offspring_diversity_retries
        self.fitness_sharing = fitness_sharing
        self.sharing_radius = sharing_radius
        self.sharing_alpha = sharing_alpha
        self.final_selection = final_selection
        self.final_selection_top_k = final_selection_top_k
        self.final_selection_cv = final_selection_cv
        self._validate_final_selection_top_k()

        _validate_parallel_backend(self.parallel_backend)
        _validate_error_score(self.error_score)
        _validate_population_initializer(self.population_initializer, feature_selection=True)
        self._validate_optimization_settings()

        # added new check for whether the estimator is compatible with scikit-learn
        if not (
            _is_classifier(self.estimator)
            or _is_regressor(self.estimator)
            or _is_outlier_detector(self.estimator)
        ):
            raise ValueError(
                f"{self.estimator} is not a valid Sklearn classifier, regressor, or outlier detector"
            )

        self.criteria_sign = _get_criteria_sign(self.criteria)

    def _register(self):
        from ._deap_backend import register_features

        self.criteria_sign = _get_criteria_sign(self.criteria)
        register_features(self)

    def _initialize_population(self):
        from ._deap_backend import initialize_population

        return initialize_population(self, initialize_feature_population)

    def select(self, population, k):
        from ._deap_backend import select_tournament

        return select_tournament(self, population, k)

    def _repair_individual(self, individual):
        mask_type = type(individual)
        if mask_type.__setitem__ is list.__setitem__ and mask_type.__iter__ is list.__iter__:
            # Ordinary lists and DEAP masks can replace their integer genes in bulk.
            individual[:] = [1 if value else 0 for value in individual]
        else:
            for index, value in enumerate(individual):
                individual[index] = 1 if value else 0

        count = sum(individual)
        max_features = getattr(self, "max_features", None)
        if max_features and count > max_features:
            selected = [index for index, value in enumerate(individual) if value]
            random.shuffle(selected)
            for index in selected[max_features:]:
                individual[index] = 0
        elif count == 0:
            individual[random.randrange(0, len(individual))] = 1

        return individual

    def _new_feature_individual(self):
        return self._repair_individual(self.toolbox.individual_raw())

    def mate(self, individual_1, individual_2):
        offspring_1, offspring_2 = self.toolbox.mate_raw(individual_1, individual_2)
        self._repair_individual(offspring_1)
        self._repair_individual(offspring_2)
        return offspring_1, offspring_2

    def mutate(self, individual):
        (mutated,) = self.toolbox.mutate_raw(individual)
        self._repair_individual(mutated)
        return (mutated,)

    def _individual_key(self, individual):
        return tuple(individual)

    def evaluate_population(self, individuals):
        for individual in individuals:
            self._repair_individual(individual)
        return _evaluate_population_batch(self, individuals, "current_generation_features")

    def _build_feature_evaluation_record(self, current_generation_params, cv_results):
        cv_scores = cv_results[f"test_{self.refit_metric}"]
        score = score_mean(cv_scores)

        current_generation_params["score"] = score
        current_generation_params["cv_scores"] = cv_scores
        current_generation_params["fit_time"] = cv_results["fit_time"]
        current_generation_params["score_time"] = cv_results["score_time"]

        for metric in self.metrics_list:
            current_generation_params[f"test_{metric}"] = cv_results[f"test_{metric}"]

            if self.return_train_score:
                current_generation_params[f"train_{metric}"] = cv_results[f"train_{metric}"]

        return score, current_generation_params

    def _penalized_feature_cv_results(self, score):
        return {
            "fit_time": np.zeros(self.n_splits_),
            "score_time": np.zeros(self.n_splits_),
            **self._constant_cv_scores(score, self.n_splits_, self.return_train_score),
        }

    def _select_feature_columns(self, mask):
        if self.preserve_dataframe:
            return self.X_.iloc[:, mask]
        return self.X_[:, mask]

    def _evaluate_individual(self, individual, n_jobs=None, *, from_cache_key=False):
        # Both evaluation entry points repair the mask before cache lookup.
        if from_cache_key:
            bool_individual = np.frombuffer(bytes(individual), dtype=bool).copy()
        else:
            bool_individual = np.array(individual, dtype=bool)

        current_generation_params = {"features": bool_individual}

        n_selected_features = np.count_nonzero(bool_individual)

        max_features = getattr(self, "max_features", None)

        if max_features and (n_selected_features > max_features or n_selected_features == 0):
            score = -self.criteria_sign * 100000
            cv_results = self._penalized_feature_cv_results(score)
            _, current_generation_params = self._build_feature_evaluation_record(
                current_generation_params, cv_results
            )

            fitness_result = [score, n_selected_features]

            return fitness_result, current_generation_params, False, True

        # cross_validate clones the estimator for each fold. Logging still needs
        # its own candidate estimator.
        local_estimator = self.estimator if self.log_config is None else clone(self.estimator)

        # Use standard cross_validate for all estimator types. Only X is
        # feature-sliced; fit params such as sample_weight stay aligned to
        # samples and are sliced per CV fold by scikit-learn.
        cv_results = self._cross_validate_candidate(
            local_estimator,
            self._select_feature_columns(bool_individual),
            self._cv_splits,
            n_jobs=n_jobs,
            return_train_score=self.return_train_score,
        )

        score, current_generation_params = self._build_feature_evaluation_record(
            current_generation_params, cv_results
        )

        # Uses the log config to save in remote log server (e.g MLflow)
        if self.log_config is not None and cv_results["fit_time"] is not None:
            self.log_config.create_run(
                parameters=current_generation_params,
                score=score,
                estimator=local_estimator,
            )

        fitness_result = [_evolution_score(score, self.criteria_sign), n_selected_features]

        return fitness_result, current_generation_params, True, False

    def evaluate(self, individual):
        """
        Compute the cross-validation scores and record the logbook and mlflow (if specified)
        Parameters
        ----------
        individual: Individual object
            The individual (set of features) that is being evaluated

        Returns
        -------
        fitness: List
            Returns a list with two values.
            The first one is the corresponding to the cv-score
            The second one is the number of features selected

        """

        # Convert the individual to a tuple to use as a key in the cache
        self._repair_individual(individual)
        individual_key = self._individual_key(individual) if self.use_cache else None

        # Check if the individual has already been evaluated
        if self.use_cache and individual_key in self.fitness_cache:
            cached_result = self.fitness_cache[individual_key]
            # Ensure the logbook is updated even if the individual is cached
            record_candidate(self, cached_result["current_generation_features"])
            _record_fit_stats(self, evaluated=1, cache_hits=1)
            return cached_result["fitness"]

        candidate_n_jobs = self.n_jobs if self.parallel_backend == "cv" else 1
        (
            fitness_result,
            current_generation_params,
            used_cv,
            skipped_invalid,
        ) = self._evaluate_individual(
            individual,
            n_jobs=candidate_n_jobs,
        )
        current_generation_params = record_candidate(
            self,
            current_generation_params,
        )

        if self.use_cache:
            # Store the fitness result and the current generation features in the cache
            self.fitness_cache[individual_key] = {
                "fitness": fitness_result,
                "current_generation_features": current_generation_params,
            }

        _record_fit_stats(
            self,
            evaluated=1,
            unique=1,
            cv_calls=int(used_cv),
            skipped=int(skipped_invalid),
        )

        return fitness_result

    def _archived_feature_row(self, records, archived_score, mask):
        means = self.cv_results_[f"mean_test_{self.refit_metric}"]
        finite = np.isfinite(archived_score)
        for index, row in enumerate(records):
            score = row.get("score", means[index])
            # The archive treats every non-finite score as the same worst fitness.
            if (score == archived_score or (not finite and not np.isfinite(score))) and (
                np.array_equal(row["features"], mask)
            ):
                return index

        # Older checkpoints can have an archived score without its exact row.
        # Keep the selected mask and use its best available search result.
        ranks = self.cv_results_[f"rank_test_{self.refit_metric}"]
        matching = (
            index for index, row in enumerate(records) if np.array_equal(row["features"], mask)
        )
        return min(matching, key=ranks.__getitem__, default=None)

    def fit(self, X, y=None, callbacks=None, groups=None, **fit_params):
        """
        Main method of GAFeatureSelectionCV, starts the optimization
        procedure with to find the best features set

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit, such as a list, array, or DataFrame.
            Use a compatible estimator or encoder for categorical columns.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
            Pass a target matrix for an estimator that supports multi-output
            or multilabel targets. Omit y for unlabeled outlier detection.
        callbacks: list or callable
            One or a list of the callbacks methods available in
            :class:`~sklearn_genetic.callbacks`.
            The callback is evaluated after fitting the estimators from the generation 1.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test sets. Only used in conjunction with a "Group" cv
            instance such as :class:`~sklearn.model_selection.GroupKFold`.
        **fit_params : dict of str -> object
            Parameters for candidate and finalist cross-validation and final
            refit, such as ``sample_weight``. With metadata routing enabled,
            scikit-learn routes them to the estimator and scorers according
            to their requests. Only ``X`` is sliced by feature, and
            sample-aligned parameters are sliced per CV fold. The final
            refit receives the estimator's fit parameters.
        """

        self._clear_fit_details()
        _check_required_target(self.estimator, y)

        _check_feature_selection_input(self.estimator)
        preserve_frame = self.preserve_dataframe
        if not isinstance(preserve_frame, (bool, np.bool_)):
            raise ValueError("preserve_dataframe must be a boolean")
        if preserve_frame:
            _check_feature_dataframe(X)
        _check_feature_names(self, X, reset=True)
        # Keep compressed inputs in place; other sparse formats need column slicing.
        sparse_formats = ("csc", "csr")
        if y is None:
            self.X_, self.y_ = X, None
        else:
            accepts_matrix, matrix_required = _feature_target_format(self.estimator)
            if preserve_frame:
                self.X_, self.y_ = X, _check_y(y, multi_output=accepts_matrix, estimator=self)
                check_consistent_length(self.X_, self.y_)
            else:
                self.X_, self.y_ = check_X_y(
                    X,
                    y,
                    accept_sparse=sparse_formats,
                    multi_output=accepts_matrix,
                    **_feature_input_checks(self),
                )
            # Preserve the usual column-vector conversion for single-output fits.
            if (
                accepts_matrix
                and not matrix_required
                and self.y_.ndim == 2
                and self.y_.shape[1] == 1
                and not issparse(self.y_)
            ):
                self.y_ = column_or_1d(self.y_, warn=True)
        self.groups_ = groups
        self._fit_params = fit_params
        _check_fit_params_supported(self.estimator, fit_params)

        if not preserve_frame and y is None:
            checks = _feature_input_checks(self)
            if _is_outlier_detector(self.estimator):
                # Let the wrapped detector decide whether non-finite values are valid.
                checks[_FINITE_CHECK] = False
            self.X_ = check_array(X, accept_sparse=sparse_formats, **checks)

        self.n_features = self.X_.shape[1]

        # Make sure the callbacks are valid
        self.callbacks = check_callback(callbacks)
        optimizer = self._validate_backend()

        checkpoint, context = self._prepare_checkpoint(optimizer)
        legacy_checkpoint = checkpoint if optimizer is None else None
        if (
            legacy_checkpoint is not None
            and legacy_checkpoint["estimator_state"].get("preserve_dataframe", False)
            != preserve_frame
        ):
            raise ValueError(
                "Checkpoint uses a different preserve_dataframe setting. Use a new checkpoint path."
            )
        runtime = _restore_legacy_checkpoint(self, legacy_checkpoint)
        if legacy_checkpoint is not None:
            _check_feature_selection_input(self.estimator)
        restored_adapter_state = runtime.get("adapter_state")
        if optimizer is None:
            _validate_search_counts(self, "DEAP")
        self._n_iterations = int(self.generations) + 1

        _validate_parallel_backend(self.parallel_backend, n_jobs=self.n_jobs)
        _validate_error_score(self.error_score)
        _validate_population_initializer(self.population_initializer, feature_selection=True)
        self.features_proportion = None
        max_features = getattr(self, "max_features", None)
        if max_features is not None:
            if (
                isinstance(max_features, (bool, np.bool_))
                or not isinstance(max_features, Integral)
                or not 1 <= max_features <= self.n_features
            ):
                raise ValueError("max_features must be an integer between 1 and n_features or None")
            self.features_proportion = max_features / self.n_features

        # Seed after the checkpoint (if any) is loaded: ``self.__dict__.update()``
        # above may have just replaced ``self.random_state`` with the value from
        # a resumed run's checkpoint, and seeding before that point would use the
        # pre-resume value instead (see #299).
        _seed_global_rngs(self.random_state)

        self._validate_optimization_settings()
        self._validate_final_selection_mode()
        self._configure_adapters()
        if restored_adapter_state is not None:
            # Restore the crossover/mutation adapter step counters so the
            # schedule continues from where the previous run left off
            # instead of restarting from step 0 (see adapter checkpoint).
            self.crossover_adapter.load_state_dict(restored_adapter_state["crossover"])
            self.mutation_adapter.load_state_dict(restored_adapter_state["mutation"])
        elif legacy_checkpoint is None:
            self.fitness_cache = {}
            _reset_adapters(self)

        # Preserve cumulative counters across a resume instead of zeroing them,
        # so e.g. ``cache_hits``/``evaluated_candidates`` reflect the whole run.
        self.fit_stats_ = runtime.get("fit_stats_") or _create_fit_stats()
        self._resume_generation_log = (
            legacy_checkpoint.get("logbook") if legacy_checkpoint else None
        )

        self._set_search_scoring(checkpoint)

        self._set_search_splits(checkpoint, optimizer, runtime)

        _restore_search_rngs(runtime.get("rng_state"))
        self.criteria_sign = _get_criteria_sign(self.criteria)
        log, n_gen = self._run_optimizer(optimizer, runtime, checkpoint, context)

        # Update the _n_iterations value as the algorithm could stop earlier due a callback
        self._n_iterations = n_gen

        records = candidate_records(self)
        self.cv_results_ = create_feature_selection_cv_results_(
            records=records,
            return_train_score=self.return_train_score,
            metrics=self.metrics_list,
            greater_is_better=self.criteria == "max",
            unique_candidates=self.fit_stats_.get("unique_candidates"),
        )

        self._set_search_history(log)

        best_features = np.array(self._hof[0], dtype=bool)
        run_finalists = self.refit and self.final_selection
        index = None
        if not run_finalists:
            archived_score = (
                self._hof[0].fitness.values[0]
                if optimizer is None
                else self._feature_elites[0][0][1][0]
            )
            index = self._archived_feature_row(records, archived_score, best_features)
            if index is not None and records[index]["fit_time"] is None:
                index = int(self._top_candidate_indices()[0])
                best_features = np.array(self.cv_results_["features"][index], dtype=bool)

        if run_finalists or callable(self.refit):
            self.best_index_, best_score, selected = self._select_final_candidate(
                allow_fold_reuse=legacy_checkpoint is None
                or (context is not None and self._copy_cv_indices),
                original_best_index=index,
            )
            if best_score is not None:
                self.best_score_ = best_score
            best_features = np.asarray(selected["features"], dtype=bool)
        elif index is not None:
            self.best_index_ = index
            self.best_score_ = float(self.cv_results_[f"mean_test_{self.refit_metric}"][index])

        if self.refit:
            best_estimator = clone(self.estimator)

            refit_params = self._refit_metadata(best_estimator)
            targets = self._target_args(self.y_)
            refit_start_time = time.monotonic()
            best_estimator.fit(
                self._select_feature_columns(best_features), *targets, **refit_params
            )
            self.refit_time_ = time.monotonic() - refit_start_time
            self.fit_counts_["refit_fits"] = 1

            self.best_estimator_ = self.estimator_ = best_estimator

        if optimizer is None:
            self.hof = self._hof

        self.best_features_ = self.support_ = best_features
        self._preserved_dataframe_ = bool(preserve_frame)
        return self

    def _more_tags(self):
        tags = {"allow_nan": _allows_nan(self.estimator)}
        if validate_data is None:
            first = _feature_input_estimator(self.estimator)
            if first is not None:
                input_types = _safe_tags(first, key="X_types")
                if "2darray" in input_types:
                    text_types = [kind for kind in ("categorical", "string") if kind in input_types]
                    tags["X_types"] = ["2darray", *text_types]
            target = _safe_tags(_feature_target_estimator(self.estimator))
            tags["requires_y"] = _safe_tags(self.estimator, key="requires_y")
            tags["multioutput"] = target["multioutput"]
            tags["multioutput_only"] = target["multioutput_only"]
            if self.refit:
                tags["multilabel"] = target["multilabel"]
        return tags

    def get_metadata_routing(self):
        """Expose metadata requests for fitting, feature selection and scoring."""
        scorer, _ = self._configured_scorer()
        request = self._get_metadata_request()
        splitters = (self.cv,)
        if self.refit and self.final_selection:
            splitters += (self.final_selection_cv,)
        if request.fit.requests.get("groups") is None and any(
            get_routing_for_object(cv).consumes("split", ("groups",)) for cv in splitters
        ):
            # fit consumes groups itself to build the common folds before candidate scoring.
            request.fit.add_request(param="groups", alias=True)
        return self._fit_metadata_router(self.estimator, scorer).add_self_request(request)

    @property
    def _estimator_type(self):
        if not self.refit:
            return None
        if validate_data is None:
            return getattr(self.estimator, "_estimator_type", None)
        return get_tags(self.estimator).estimator_type

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        wrapped = get_tags(self.estimator)
        first = _feature_input_estimator(self.estimator)
        inputs = (
            wrapped.input_tags
            if first is None or first is self.estimator
            else get_tags(first).input_tags
        )
        final = _feature_target_estimator(self.estimator)
        target = wrapped.target_tags if final is self.estimator else get_tags(final).target_tags
        tags.input_tags.allow_nan = wrapped.input_tags.allow_nan
        tags.input_tags.sparse = wrapped.input_tags.sparse and not self.preserve_dataframe
        tags.input_tags.categorical = inputs.two_d_array and inputs.categorical
        tags.input_tags.string = inputs.two_d_array and inputs.string
        tags.target_tags.required = wrapped.target_tags.required
        tags.target_tags.multi_output = target.multi_output
        tags.target_tags.single_output = target.single_output
        tags.target_tags.two_d_labels = target.two_d_labels
        if self.refit:
            tags.estimator_type = wrapped.estimator_type
            tags.classifier_tags = deepcopy(wrapped.classifier_tags)
            tags.regressor_tags = deepcopy(wrapped.regressor_tags)
        return tags

    def __sklearn_is_fitted__(self):
        return "best_features_" in vars(self)

    @property
    def n_features_(self):
        """Number of selected features in the support mask."""
        return int(np.count_nonzero(self.support_))

    @property
    def classes_(self):
        """Class labels of the refitted classifier."""
        if not self.refit:
            raise AttributeError("classes_ requires refit=True")
        return self.best_estimator_.classes_

    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        if not self.__sklearn_is_fitted__():
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(self.__class__.__name__)
            )

        return self.n_features

    def _get_support_mask(self):
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                "This GAFeatureSelectionCV instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )
        return self.best_features_

    def inverse_transform(self, X):
        """Restore original column positions, filling removed columns with zeros.

        Dense inputs can contain NaN when the wrapped estimator supports it.
        """
        if issparse(X) or not _allows_nan(self):
            return super().inverse_transform(X)

        support = self.get_support()
        selected = check_array(X, dtype=None, **{_FINITE_CHECK: "allow-nan"})
        if selected.shape[1] != support.sum():
            raise ValueError("X has a different shape than during fitting.")
        restored = np.zeros((selected.shape[0], support.size), dtype=selected.dtype)
        restored[:, support] = selected
        return restored

    def _transform_for_prediction(self, X):
        if not self._fitted:
            raise NotFittedError(
                f"This GAFeatureSelectionCV instance is not fitted yet "
                f"or used refit=False. Call 'fit' with appropriate "
                f"arguments before using this estimator."
            )
        if getattr(self, "_preserved_dataframe_", False):
            _check_feature_dataframe(X)
            if validate_data is None:
                self._validate_data(X, reset=False, cast_to_ndarray=False)
            else:
                validate_data(self, X, reset=False, skip_check_array=True)
            return self._transform(X)
        # Keep compressed inputs in their existing format for the wrapped estimator.
        finite = "allow-nan" if _allows_nan(self) else True
        options = {
            "reset": False,
            "dtype": None,
            "accept_sparse": ("csr", "csc"),
            _FINITE_CHECK: finite,
        }
        if validate_data is None:
            X = self._validate_data(X, **options)
        else:
            X = validate_data(self, X, **options)
        return self._transform(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found features.
       Only available if ``refit=True`` and the underlying estimator supports
       ``decision_function``.

       Parameters
       ----------
       X : indexable, length n_samples
           Must fulfill the input assumptions of the
           underlying estimator.

       Returns
       -------
       y_score : ndarray of shape (n_samples,) or (n_samples, n_classes) \
               or (n_samples, n_classes * (n_classes-1) / 2)
           Result of the decision function for `X` based on the estimator with
           the best found parameters.
       """
        X = self._transform_for_prediction(X)
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Call predict on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        X = self._transform_for_prediction(X)
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray or list of ndarray
            Predicted class log-probabilities for `X` based on the estimator
            with the best found parameters, in the fitted `classes_` order.
            Multi-output classifiers can return one array per output.
        """
        X = self._transform_for_prediction(X)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray or list of ndarray
            Predicted class probabilities for `X` based on the estimator with
            the best found parameters, in the fitted `classes_` order.
            Multi-output classifiers can return one array per output.
        """
        X = self._transform_for_prediction(X)
        return self.best_estimator_.predict_proba(X)

    def _can_score(self):
        return (
            self.scoring is not None
            or _is_outlier_detector(self.estimator)
            or _estimator_has("score")(self)
        )

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        """Return per-sample scores from the refitted estimator on selected features."""
        X = self._transform_for_prediction(X)
        return self.best_estimator_.score_samples(X)

    @available_if(_can_score)
    def score(self, X, y=None, *, sample_weight=None):
        """Return the score on the given data, if the estimator has been refit.
        This uses the configured scorer, including the default anomaly scorer
        for an outlier detector with ``scoring=None``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        sample_weight : array-like of shape (n_samples,), default=None
            Weights for the samples being scored. Passed to the selected
            scorer when supplied.

        Returns
        -------
        score : float
            The score for the selected features.
        """
        X = self._transform_for_prediction(X)
        scorer = self.scorer_[self.refit_metric] if isinstance(self.scorer_, dict) else self.scorer_
        targets = self._target_args(y)
        if sample_weight is None:
            return scorer(self.best_estimator_, X, *targets)
        return scorer(self.best_estimator_, X, *targets, sample_weight=sample_weight)


# helpers

_NATIVE_PARAMETER_EVALUATOR = GASearchCV._evaluate_individual
_NATIVE_PARAMETER_REPAIR = GASearchCV._repair_individual
_NATIVE_PARAMETER_REPAIR_VALUE = GASearchCV._repair_value
_NATIVE_INTEGER_CLIP = Integer._round_and_clip
_NATIVE_INTEGER_SAMPLE = Integer.sample


def _uses_prepared_integer_evaluation(estimator):
    # Native integer repair always returns an in-range Python integer.
    return (
        type(estimator) is GASearchCV
        and getattr(estimator._evaluate_individual, "__func__", None) is _NATIVE_PARAMETER_EVALUATOR
        and getattr(estimator._repair_individual, "__func__", None) is _NATIVE_PARAMETER_REPAIR
        and estimator._repair_value is _NATIVE_PARAMETER_REPAIR_VALUE
        and all(
            type(dimension) is Integer
            and type(dimension.lower) is type(dimension.upper) is int
            and getattr(dimension._round_and_clip, "__func__", None) is _NATIVE_INTEGER_CLIP
            and getattr(dimension.sample, "__func__", None) is _NATIVE_INTEGER_SAMPLE
            for dimension in estimator.space.param_grid.values()
        )
    )


# Preserve class and instance overrides when dispatching validated backend rows.
_NATIVE_FEATURE_EVALUATION = GAFeatureSelectionCV.evaluate_population
_NATIVE_FEATURE_INDIVIDUAL_EVALUATOR = GAFeatureSelectionCV._evaluate_individual
_NATIVE_FEATURE_KEY = GAFeatureSelectionCV._individual_key
_NATIVE_FEATURE_REPAIR = GAFeatureSelectionCV._repair_individual
_NATIVE_FEATURE_INITIALIZER = initialize_feature_population
_NATIVE_FEATURE_SAMPLE = GAFeatureSelectionCV._new_feature_individual


def _uses_native_feature_evaluation(estimator):
    return (
        type(estimator) is GAFeatureSelectionCV
        and getattr(estimator.evaluate_population, "__func__", None) is _NATIVE_FEATURE_EVALUATION
        and getattr(estimator._repair_individual, "__func__", None) is _NATIVE_FEATURE_REPAIR
    )


def _uses_cached_feature_mask(estimator, evaluate):
    return (
        _uses_native_feature_evaluation(estimator)
        and getattr(evaluate, "__func__", None) is _NATIVE_FEATURE_INDIVIDUAL_EVALUATOR
        and getattr(estimator._individual_key, "__func__", None) is _NATIVE_FEATURE_KEY
        and issparse(estimator.X_)
    )


def _uses_native_feature_initialization(estimator, initializer):
    # Smart, spread and warm-start masks are built within the feature bounds.
    # Random initialization uses a sampler that repairs each mask on creation.
    return (
        initializer is _NATIVE_FEATURE_INITIALIZER
        and _uses_native_feature_evaluation(estimator)
        and (
            estimator.population_initializer != "random"
            or getattr(estimator._new_feature_individual, "__func__", None)
            is _NATIVE_FEATURE_SAMPLE
        )
    )


def ind_fitness_values(ind):
    return ind.fitness.values
