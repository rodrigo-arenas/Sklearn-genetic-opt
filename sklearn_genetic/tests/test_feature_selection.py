import pytest
from array import array
from functools import partial
import warnings
from scipy import sparse
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    OutlierMixin,
    TransformerMixin,
    clone,
    is_classifier,
    is_regressor,
    is_outlier_detector,
)
from sklearn.datasets import load_iris, load_diabetes
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
import numpy as np

from .. import (
    EvolutionConfig,
    GAFeatureSelectionCV,
    OptimizationConfig,
    PopulationConfig,
    RuntimeConfig,
)
from .. import genetic_search
from ..callbacks import (
    ThresholdStopping,
    DeltaThreshold,
    ConsecutiveStopping,
    TimerStopping,
    ProgressBar,
)
from ..schedules import ExponentialAdapter, InverseAdapter

data = load_iris()
label_names = data["target_names"]
y = data["target"]
X = data["data"]

noise = np.random.uniform(1, 4, size=(X.shape[0], 10))

X = np.hstack((X, noise))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def _fit_feature_name_selector(X, y, refit):
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=1,
        verbose=False,
        n_jobs=1,
        random_state=42,
        refit=refit,
    )
    return selector.fit(X, y)


@pytest.mark.parametrize("copies, population, max_features", [(65, 8, 8), (5, 24, 1)])
def test_wide_mask_diversity_matches_population_pair_disagreements(
    copies, population, max_features
):
    X, y = load_iris(return_X_y=True)
    data = np.tile(X, (1, copies))
    selector = GAFeatureSelectionCV(
        DummyClassifier(),
        population_size=population,
        generations=0,
        cv=2,
        max_features=max_features,
        use_cache=False,
        random_state=31,
        n_jobs=1,
        verbose=False,
    ).fit(data, y)
    masks = np.asarray(selector.cv_results_["features"])
    n = len(masks)
    assert n == selector.population_size
    unequal = sum(
        np.not_equal(left, right) for i, left in enumerate(masks) for right in masks[i + 1 :]
    )
    expected = np.mean(1 - (n * (n - 1) - 2 * unequal) / (n * (n - 1)))
    assert selector.history["genotype_diversity"] == [expected]
    assert selector.history["unique_individuals"] == [len(set(map(tuple, masks)))]


def test_get_feature_names_out_requires_fitted_selector():
    selector = GAFeatureSelectionCV(DecisionTreeClassifier())

    with pytest.raises(NotFittedError, match="not fitted"):
        selector.get_feature_names_out()


@pytest.mark.parametrize("refit", [True, False])
def test_get_feature_names_out_generates_numpy_feature_names(refit):
    iris = load_iris()
    selector = _fit_feature_name_selector(iris.data, iris.target, refit)
    input_features = np.asarray([f"x{index}" for index in range(iris.data.shape[1])])

    assert not hasattr(selector, "feature_names_in_")
    np.testing.assert_array_equal(
        selector.get_feature_names_out(),
        input_features[selector.get_support()],
    )

    with pytest.raises(ValueError, match="input_features should have length equal"):
        selector.get_feature_names_out(["first", "second"])


@pytest.mark.parametrize("refit", [True, False])
def test_get_feature_names_out_preserves_dataframe_feature_names(refit):
    iris = load_iris()
    input_features = np.asarray(
        ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        dtype=object,
    )
    X_frame = pd.DataFrame(iris.data, columns=input_features)
    selector = _fit_feature_name_selector(X_frame, iris.target, refit)
    expected = input_features[selector.get_support()]

    np.testing.assert_array_equal(selector.feature_names_in_, input_features)
    np.testing.assert_array_equal(selector.get_feature_names_out(), expected)
    np.testing.assert_array_equal(selector.get_feature_names_out(input_features), expected)

    mismatched = input_features.copy()
    mismatched[0] = "different"
    with pytest.raises(ValueError, match="input_features is not equal to feature_names_in_"):
        selector.get_feature_names_out(mismatched)


def test_failed_fit_with_mixed_column_types_discards_previous_names():
    iris = load_iris(as_frame=True)
    selector = _fit_feature_name_selector(iris.data, iris.target, refit=False)
    invalid = iris.data.rename(columns={iris.data.columns[0]: 0})
    with pytest.raises(TypeError, match="feature names"):
        selector.fit(invalid, iris.target)
    assert not hasattr(selector, "feature_names_in_")

    updated = iris.data.add_prefix("new_")
    selector.fit(updated, iris.target)
    np.testing.assert_array_equal(selector.feature_names_in_, updated.columns)


def test_named_transform_output_preserves_prediction_input_contract():
    iris = load_iris(as_frame=True)
    selector = _fit_feature_name_selector(iris.data, iris.target, refit=True)
    positions = selector.get_support()
    expected_input = iris.data.to_numpy()[:, positions]
    selector.set_output(transform="pandas")

    pd.testing.assert_frame_equal(selector.transform(iris.data), iris.data.iloc[:, positions])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="X has feature names")
        np.testing.assert_array_equal(
            selector.predict(iris.data), selector.best_estimator_.predict(expected_input)
        )
        np.testing.assert_array_equal(
            selector.predict_proba(iris.data),
            selector.best_estimator_.predict_proba(expected_input),
        )
        assert selector.score(iris.data, iris.target) == selector.best_estimator_.score(
            expected_input, iris.target
        )
    with pytest.raises(ValueError, match="feature names should match"):
        selector.predict(iris.data.iloc[:, ::-1])


def test_sparse_prediction_remains_available_with_named_transform_output():
    X, y = load_iris(return_X_y=True)
    X = sparse.csr_matrix(X)
    selector = _fit_feature_name_selector(X, y, refit=True)
    expected = selector.predict(X)
    selector.set_output(transform="pandas")

    np.testing.assert_array_equal(selector.predict(X), expected)
    with pytest.raises(ValueError, match="features"):
        selector.predict(X[:, :-1])


@pytest.mark.parametrize(
    ("layout", "supports_sparse"),
    [("pandas", True), ("pandas", False), ("csr", True), ("csc", True)],
)
def test_selector_pipeline_preserves_columns_and_reports_sparse_support(layout, supports_sparse):
    from packaging.version import Version
    from sklearn import __version__ as sklearn_version
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline

    iris = load_iris(as_frame=True)
    X = iris.data
    X.index = [f"sample-{position}" for position in range(len(X))]
    if layout != "pandas":
        X = sparse.csr_matrix(X).asformat(layout)
    model = DecisionTreeClassifier(random_state=0) if supports_sparse else GaussianNB()
    selector = GAFeatureSelectionCV(
        model,
        cv=2,
        refit=False,
        max_features=2,
        population_size=4,
        generations=0,
        n_jobs=1,
        random_state=0,
        verbose=False,
        error_score="raise",
    )
    if layout == "pandas":
        selector.set_output(transform="pandas")
    pipeline = make_pipeline(selector, DecisionTreeClassifier(random_state=0))
    pipeline.fit(X, iris.target)
    fitted_selector, classifier = [step for _, step in pipeline.steps]
    columns = fitted_selector.get_support(indices=True)
    transformed = fitted_selector.transform(X)

    if layout == "pandas":
        pd.testing.assert_frame_equal(transformed, X.iloc[:, columns])
        np.testing.assert_array_equal(classifier.feature_names_in_, transformed.columns)
    else:
        assert transformed.format == "csr"
        np.testing.assert_array_equal(transformed.toarray(), X[:, columns].toarray())
        np.testing.assert_array_equal(
            fitted_selector.get_feature_names_out(), [f"x{index}" for index in columns]
        )
    np.testing.assert_array_equal(pipeline.predict(X), classifier.predict(transformed))

    if genetic_search.validate_data is not None:
        advertised = genetic_search.get_tags(model).input_tags.sparse
        assert genetic_search.get_tags(fitted_selector).input_tags.sparse is advertised
        # Pipeline gained sparse tag propagation in scikit-learn 1.6.1.
        if Version(sklearn_version) >= Version("1.6.1"):
            final_support = genetic_search.get_tags(classifier).input_tags.sparse
            assert genetic_search.get_tags(pipeline).input_tags.sparse is (
                advertised and final_support
            )


@pytest.mark.parametrize("as_dataframe", [False, True])
@pytest.mark.parametrize("encoder_name", ["onehot", "ordinal"])
def test_feature_selection_preserves_categories_for_pipeline_encoder(as_dataframe, encoder_name):
    iris = load_iris(as_frame=True)
    X = np.where(iris.data.to_numpy() < [5.8, 3.0, 3.8, 1.2], "low", "high")
    X_train, X_test, y_train, y_test = train_test_split(
        X, iris.target, test_size=0.25, random_state=0, stratify=iris.target
    )
    X_test[0] = "new"
    if as_dataframe:
        X_train = pd.DataFrame(X_train, columns=iris.feature_names)
        X_test = pd.DataFrame(X_test, columns=iris.feature_names)
    encoder = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if encoder_name == "onehot"
        else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )
    pipeline = make_pipeline("passthrough", encoder, DecisionTreeClassifier(random_state=0))
    selector = GAFeatureSelectionCV(
        pipeline,
        scoring="accuracy",
        cv=2,
        population_size=4,
        generations=0,
        max_features=3,
        verbose=False,
        n_jobs=1,
        random_state=0,
        error_score="raise",
    )
    if as_dataframe:
        selector.set_output(transform="pandas")
    selector.fit(X_train, y_train)

    selected = selector.transform(X_test)
    columns = selector.get_support(indices=True)
    if as_dataframe:
        pd.testing.assert_frame_equal(selected, X_test.iloc[:, columns])
        np.testing.assert_array_equal(selector.get_feature_names_out(), X_test.columns[columns])
        selected = selected.to_numpy()
    else:
        np.testing.assert_array_equal(selected, X_test[:, columns])
    predictions = selector.predict(X_test)
    np.testing.assert_array_equal(predictions, selector.best_estimator_.predict(selected))
    assert selector.score(X_test, y_test) == accuracy_score(y_test, predictions)
    if genetic_search.validate_data is None:
        assert "categorical" in selector._get_tags()["X_types"]
    else:
        assert genetic_search.get_tags(selector).input_tags.categorical


@pytest.mark.parametrize("kind", ["kernel", "distances", "pipeline", "outlier"])
def test_feature_selection_rejects_precomputed_pairwise_input_before_cv(kind, monkeypatch):
    X, y = load_iris(return_X_y=True)
    if kind in ("distances", "outlier"):
        pairwise = euclidean_distances(X)
        estimator = KNeighborsClassifier(metric="precomputed")
        if kind == "outlier":
            estimator = LocalOutlierFactor(metric="precomputed", novelty=True)
            y = None
    else:
        pairwise = rbf_kernel(X, gamma=0.2)
        estimator = SVC(kernel="precomputed")
        if kind == "pipeline":
            estimator = make_pipeline("passthrough", estimator)

    cv_calls = []
    original = genetic_search.cross_validate

    def count_cv(*args, **kwargs):
        cv_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(genetic_search, "cross_validate", count_cv)
    selector = GAFeatureSelectionCV(
        estimator,
        cv=2,
        population_size=4,
        generations=0,
        max_features=2,
        n_jobs=1,
        random_state=0,
        verbose=False,
        error_score="raise",
    )

    with pytest.raises(ValueError, match="GAFeatureSelectionCV requires original feature columns"):
        selector.fit(pairwise, y)
    assert not cv_calls
    assert not hasattr(selector, "fit_counts_")


def test_feature_selection_allows_kernel_computed_inside_pipeline():
    class RBFKernel(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            self.samples_ = X.copy()
            return self

        def transform(self, X):
            return rbf_kernel(X, self.samples_, gamma=0.2)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )
    selector = GAFeatureSelectionCV(
        make_pipeline(RBFKernel(), SVC(kernel="precomputed")),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=0,
        max_features=3,
        n_jobs=1,
        random_state=0,
        verbose=False,
        error_score="raise",
    ).fit(X_train, y_train)

    native = SVC(gamma=0.2)
    selected_train = selector.transform(X_train)
    selected_test = selector.transform(X_test)
    reference = cross_val_score(native, selected_train, y_train, scoring="accuracy", cv=2)
    assert selector.best_score_ == pytest.approx(reference.mean())
    native.fit(selected_train, y_train)
    np.testing.assert_array_equal(selector.predict(X_test), native.predict(selected_test))
    assert selector.score(X_test, y_test) == native.score(selected_test, y_test)


def test_feature_selection_checks_pairwise_estimator_from_checkpoint(tmp_path, monkeypatch):
    import pickle

    from ..callbacks import ModelCheckpoint

    X, y = load_iris(return_X_y=True)
    kernel = rbf_kernel(X[::5], gamma=0.2)
    targets = y[::5]
    checkpoint = ModelCheckpoint(tmp_path / "pairwise.pkl")
    selector = GAFeatureSelectionCV(
        SVC(kernel="linear"),
        cv=2,
        population_size=4,
        generations=0,
        max_features=2,
        n_jobs=1,
        random_state=0,
        verbose=False,
        error_score="raise",
    )
    clone(selector).fit(kernel, targets, callbacks=checkpoint)
    snapshot = checkpoint.load()
    snapshot["estimator_state"]["estimator"].set_params(kernel="precomputed")
    checkpoint.checkpoint_path.write_bytes(pickle.dumps(snapshot))

    def unexpected_cv(*args, **kwargs):
        pytest.fail("A checkpoint using pairwise input must fail before CV")

    monkeypatch.setattr(genetic_search, "cross_validate", unexpected_cv)
    with pytest.raises(ValueError, match="GAFeatureSelectionCV requires original feature columns"):
        selector.fit(kernel, targets, callbacks=checkpoint)


@pytest.mark.parametrize("sparse_format", ["csr", "csc"])
def test_inference_preserves_input_compressed_sparse_format(sparse_format):
    class FormatCheckingTree(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            self.input_format_ = X.format
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

        def predict(self, X, check_input=True):
            assert X.format == self.input_format_
            return super().predict(X, check_input=check_input)

        def predict_proba(self, X, check_input=True):
            assert X.format == self.input_format_
            return super().predict_proba(X, check_input=check_input)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )
    X_train = sparse.coo_matrix(X_train).asformat(sparse_format)
    X_test = sparse.coo_matrix(X_test).asformat(sparse_format)
    selector = GAFeatureSelectionCV(
        FormatCheckingTree(random_state=0),
        population_size=4,
        generations=0,
        cv=2,
        n_jobs=1,
        random_state=0,
        verbose=False,
        error_score="raise",
    ).fit(X_train, y_train)
    selected = X_test[:, selector.get_support()]
    estimator = selector.best_estimator_
    np.testing.assert_array_equal(selector.predict(X_test), estimator.predict(selected))
    np.testing.assert_array_equal(selector.predict_proba(X_test), estimator.predict_proba(selected))
    assert selector.score(X_test, y_test) == estimator.score(selected, y_test)
    assert selector.transform(X_test).format == "csr"
    if sparse_format == "csr":
        np.testing.assert_array_equal(selector.predict(X_test.tocoo()), estimator.predict(selected))


@pytest.mark.parametrize("as_dataframe", [False, True])
@pytest.mark.parametrize("encode_categories", [False, True])
def test_outlier_selection_accepts_array_like_without_target(as_dataframe, encode_categories):
    frame = pd.DataFrame(
        np.random.default_rng(0).normal(size=(30, 4)),
        columns=["first", "second", "third", "fourth"],
    )
    detector = IsolationForest(n_estimators=5, random_state=0, n_jobs=1)
    if encode_categories:
        frame = pd.DataFrame(
            np.where(frame.to_numpy() < 0, "low", "high"),
            columns=frame.columns,
        )
        detector = make_pipeline(OrdinalEncoder(), detector)
    selector = GAFeatureSelectionCV(
        detector,
        cv=2,
        population_size=2,
        generations=0,
        verbose=False,
        n_jobs=1,
        random_state=0,
    )

    X_input = frame if as_dataframe else frame.to_numpy().tolist()
    selector.fit(X_input)

    assert 1 <= selector.get_support().sum() <= frame.shape[1]
    assert selector.transform(X_input).shape == (len(frame), selector.get_support().sum())
    expected = selector.best_estimator_.score_samples(selector.transform(X_input)).mean()
    assert selector.score(X_input) == pytest.approx(expected)
    if as_dataframe:
        np.testing.assert_array_equal(selector.feature_names_in_, frame.columns.to_numpy())
        np.testing.assert_array_equal(
            selector.get_feature_names_out(), frame.columns.to_numpy()[selector.get_support()]
        )


@pytest.mark.parametrize("kind", ["regression", "multilabel", "wrapped_regression"])
def test_feature_selection_preserves_multioutput_targets(kind):
    X, iris_class = load_iris(return_X_y=True)
    if kind == "multilabel":
        y = pd.DataFrame({"first": iris_class == 0, "last": iris_class == 2})
        estimator = DecisionTreeClassifier(max_depth=3, random_state=0)
        scoring = "accuracy"
    else:
        y = np.column_stack((iris_class, 2 * X[:, 2] - X[:, 0]))
        estimator = DecisionTreeRegressor(max_depth=3, random_state=0)
        scoring = "neg_mean_squared_error"
        if kind == "wrapped_regression":
            # A multi-output-only estimator needs 2D y even with one column.
            estimator = make_pipeline(StandardScaler(), MultiOutputRegressor(estimator))
            y = y[:, :1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    selector = GAFeatureSelectionCV(
        estimator,
        scoring=scoring,
        cv=2,
        max_features=3,
        population_size=4,
        generations=0,
        verbose=False,
        n_jobs=1,
        random_state=0,
        error_score="raise",
    ).fit(X_train, y_train)

    predictions = selector.predict(X_test)
    assert predictions.shape == y_test.shape
    selected = selector.transform(X_test)
    np.testing.assert_array_equal(predictions, selector.best_estimator_.predict(selected))
    expected = cross_val_score(
        estimator, selector.transform(X_train), y_train, scoring=scoring, cv=2
    )
    assert selector.best_score_ == pytest.approx(expected.mean())
    if kind == "multilabel":
        assert len(selector.predict_proba(X_test)) == y_test.shape[1]
        assert selector.score(X_test, y_test) == accuracy_score(y_test, predictions)
    if genetic_search.validate_data is not None:
        tags = genetic_search.get_tags(selector)
        assert tags.target_tags.multi_output
        assert tags.target_tags.single_output is (kind != "wrapped_regression")
        if kind == "multilabel":
            assert tags.classifier_tags.multi_label


def test_feature_selection_preserves_single_column_target_conversion():
    X, y = load_iris(return_X_y=True)
    selector = GAFeatureSelectionCV(
        DecisionTreeRegressor(max_depth=3, random_state=0),
        cv=2,
        population_size=4,
        generations=0,
        verbose=False,
        n_jobs=1,
        random_state=0,
        error_score="raise",
    )
    with pytest.warns(DataConversionWarning, match="column-vector y"):
        selector.fit(X, y[:, None])
    assert selector.predict(X).shape == y.shape
    assert selector.y_.ndim == 1


def test_outlier_selection_score_uses_configured_scorer_on_held_out_data():
    X, labels = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels == 2, test_size=0.25, random_state=8
    )

    def scorer(estimator, X, y):
        return np.mean(-estimator.score_samples(X)[y])

    selector = GAFeatureSelectionCV(
        IsolationForest(n_estimators=5, random_state=0),
        cv=2,
        scoring=scorer,
        population_size=4,
        generations=0,
        n_jobs=1,
        random_state=0,
        verbose=False,
    )
    with pytest.raises(NotFittedError, match="not fitted"):
        selector.score(X_test, y_test)

    selector.fit(X_train, y_train)
    expected = scorer(selector.best_estimator_, selector.transform(X_test), y_test)
    assert selector.score(X_test, y_test) == pytest.approx(expected)

    selector.set_params(refit=False).fit(X_train, y_train)
    with pytest.raises(NotFittedError, match="refit=False"):
        selector.score(X_test, y_test)


@pytest.mark.parametrize("outlier", [False, True])
@pytest.mark.parametrize("refit", [True, False])
def test_feature_selection_handles_sparse_coo(outlier, refit):
    X, y = load_iris(return_X_y=True)
    model = (
        IsolationForest(n_estimators=5, random_state=0)
        if outlier
        else DecisionTreeClassifier(random_state=0)
    )
    selector = GAFeatureSelectionCV(
        model,
        cv=2,
        population_size=4,
        generations=0,
        n_jobs=1,
        random_state=0,
        verbose=False,
        refit=refit,
    )
    coo = sparse.coo_matrix(X)
    target = None if outlier else y
    compressed = coo.tocsc() if outlier else coo.tocsr()
    reference = clone(selector).fit(compressed, target)
    selector.fit(coo, target)
    assert selector.X_.format == "csc"
    assert reference.X_.format == compressed.format
    np.testing.assert_array_equal(selector.get_support(), reference.get_support())
    transformed = selector.transform(coo)
    assert sparse.issparse(transformed)
    np.testing.assert_array_equal(transformed.toarray(), X[:, reference.get_support()])
    if refit:
        np.testing.assert_array_equal(selector.predict(coo), reference.predict(compressed))
    expected_inverse = X * selector.support_
    np.testing.assert_array_equal(
        selector.inverse_transform(transformed).toarray(), expected_inverse
    )


@pytest.mark.parametrize(
    "layout, n_jobs, native_key", [("csr", 1, True), ("csc", 2, True), ("csr", 1, False)]
)
def test_wide_sparse_feature_evaluation_reuses_cache_keys(layout, n_jobs, native_key, monkeypatch):
    width = 4096
    _, y = load_iris(return_X_y=True)
    X = sparse.random(len(y), width, density=0.01, format=layout, random_state=17)
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(max_depth=2, random_state=17),
        cv=2,
        generations=0,
        population_size=4,
        max_features=8,
        random_state=17,
        n_jobs=n_jobs,
        verbose=False,
    )
    reference = clone(selector).set_params(use_cache=False).fit(X, y)
    if not native_key:
        selector._individual_key = lambda row: (tuple(row),)
    original = np.frombuffer
    decoded = []

    def tracked(buffer, *args, **kwargs):
        if type(buffer) is bytes and len(buffer) == width and kwargs.get("dtype") is bool:
            decoded.append(buffer)
        return original(buffer, *args, **kwargs)

    monkeypatch.setattr(np, "frombuffer", tracked)
    selector.fit(X, y)
    assert len(decoded) == (4 if native_key else 0)
    assert selector.fit_counts_["search_cv_calls"] == 4
    assert selector.fit_counts_ == reference.fit_counts_
    masks = selector.cv_results_["features"]
    assert all(mask.flags.owndata and mask.flags.writeable for mask in masks)
    np.testing.assert_array_equal(masks, reference.cv_results_["features"])
    np.testing.assert_array_equal(
        selector.cv_results_["mean_test_score"], reference.cv_results_["mean_test_score"]
    )
    np.testing.assert_array_equal(selector.predict(X), reference.predict(X))


@pytest.mark.parametrize("failure", ["input", "refit"])
def test_failed_feature_selection_fit_discards_the_previous_mask(failure):
    X, y = load_iris(return_X_y=True)

    class RejectFullData(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            if len(y) == 150:
                raise ValueError("rejected full-data fit")
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

    selector = GAFeatureSelectionCV(
        RejectFullData(random_state=0),
        cv=2,
        population_size=4,
        generations=0,
        n_jobs=1,
        random_state=0,
        verbose=False,
        refit=False,
        error_score="raise",
    ).fit(X, y)
    if failure == "input":
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            selector.fit(X, y[:-1])
    else:
        selector.set_params(refit=True)
        with pytest.raises(ValueError, match="rejected full-data fit"):
            selector.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(selector)
    with pytest.raises(NotFittedError):
        selector.get_support()
    assert not hasattr(selector, "n_features_in_")


@pytest.mark.parametrize("tagged", [False, True])
def test_outlier_selection_preserves_detector_nan_policy(tagged):
    class NaNTolerantOutlier(OutlierMixin, BaseEstimator):
        def _more_tags(self):
            return {"allow_nan": tagged}

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.input_tags.allow_nan = tagged
            return tags

        def fit(self, X, y=None):
            self.n_features_in_ = X.shape[1]
            self.saw_nan_ = bool(np.isnan(X).any())
            return self

        def score_samples(self, X):
            return np.nan_to_num(X, nan=0).mean(axis=1)

    frame = pd.DataFrame(
        [[np.nan, 1.0], [2.0, np.nan], [4.0, 5.0], [6.0, 7.0]],
        columns=["first", "second"],
    )
    selector = GAFeatureSelectionCV(
        NaNTolerantOutlier(),
        cv=2,
        population_size=2,
        generations=0,
        verbose=False,
        n_jobs=1,
        random_state=0,
        error_score="raise",
    )

    selector.fit(frame)

    assert selector.get_support().any()
    assert selector.best_estimator_.saw_nan_
    if tagged:
        expected = frame.to_numpy()[:, selector.get_support()]
        np.testing.assert_array_equal(selector.transform(frame), expected)
        assert selector.score(frame) == pytest.approx(
            selector.best_estimator_.score_samples(expected).mean()
        )


@pytest.mark.parametrize("backend", ["deap", "scipy", "nevergrad", "cmaes", "pymoo"])
@pytest.mark.parametrize("refit", [False, True])
def test_supervised_selection_preserves_supported_nan(backend, refit):
    if backend != "deap":
        pytest.importorskip(backend)
    frame, labels = load_iris(as_frame=True, return_X_y=True)
    for column in range(frame.shape[1]):
        frame.iloc[column::17, column] = np.nan
    train, held_out, y_train, y_test = train_test_split(
        frame, labels, test_size=0.25, random_state=31, stratify=labels
    )
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=31),
        optimizer_backend=backend,
        cv=2,
        population_size=6,
        generations=1,
        max_features=2,
        random_state=31,
        n_jobs=1,
        verbose=False,
        refit=refit,
        error_score="raise",
    ).fit(train, y_train)
    support = selector.get_support()
    expected = held_out.to_numpy()[:, support]
    assert np.isnan(expected).any()
    reduced = selector.transform(held_out)
    np.testing.assert_array_equal(reduced, expected)
    inverse = np.zeros(held_out.shape, dtype=expected.dtype)
    inverse[:, support] = expected
    np.testing.assert_array_equal(
        selector.inverse_transform(sparse.csr_matrix(reduced)).toarray(), inverse
    )
    np.testing.assert_array_equal(selector.inverse_transform(reduced), inverse)
    selector.set_output(transform="pandas")
    np.testing.assert_array_equal(selector.inverse_transform(selector.transform(held_out)), inverse)
    if refit:
        np.testing.assert_array_equal(
            selector.predict(held_out), selector.best_estimator_.predict(expected)
        )
        assert selector.score(held_out, y_test) == selector.best_estimator_.score(expected, y_test)


def test_inverse_transform_keeps_dense_validation_for_strict_estimators():
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    selector = GAFeatureSelectionCV(
        LogisticRegression(),
        cv=2,
        population_size=4,
        generations=0,
        refit=False,
        n_jobs=1,
        random_state=31,
        verbose=False,
    ).fit(X, y)
    reduced = selector.transform(X)
    reduced[0, 0] = np.nan
    with pytest.raises(ValueError, match="Input contains NaN|Input X contains NaN"):
        selector.inverse_transform(reduced)


@pytest.mark.parametrize("backend", ["deap", "scipy", "nevergrad", "cmaes", "pymoo"])
@pytest.mark.parametrize("multiclass", [False, True])
def test_feature_selector_supports_external_auc_and_stratification(backend, multiclass):
    if backend != "deap":
        pytest.importorskip(backend)
    X, y = load_iris(return_X_y=True)
    if not multiclass:
        X, y = X[y < 2], y[y < 2]
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=31),
        optimizer_backend=backend,
        max_features=2,
        cv=2,
        population_size=6,
        generations=0,
        random_state=31,
        n_jobs=1,
        verbose=False,
    )
    assert not hasattr(selector, "classes_")
    metric = "roc_auc_ovr" if multiclass else "roc_auc"
    result = cross_validate(
        selector,
        X,
        y,
        cv=3,
        scoring=metric,
        return_estimator=True,
        return_indices=True,
        error_score="raise",
    )
    folds = list(StratifiedKFold(3).split(X, y))
    expected = cross_val_score(selector, X, y, cv=folds, scoring=metric, error_score="raise")
    assert is_classifier(selector)
    np.testing.assert_array_equal(result["test_score"], expected)
    for actual, (_, test) in zip(result["indices"]["test"], folds, strict=True):
        np.testing.assert_array_equal(actual, test)
    for fitted in result["estimator"]:
        np.testing.assert_array_equal(fitted.classes_, np.unique(y))
    transformer = clone(selector).set_params(refit=False).fit(X, y)
    assert not is_classifier(transformer)
    assert not hasattr(transformer, "classes_")
    assert transformer.transform(X).shape[1] == transformer.n_features_


def test_refitted_feature_selector_is_a_regressor():
    X, y = load_diabetes(return_X_y=True)
    selector = GAFeatureSelectionCV(
        DecisionTreeRegressor(random_state=31),
        cv=2,
        population_size=4,
        generations=0,
        random_state=31,
        n_jobs=1,
        verbose=False,
    )
    assert is_regressor(selector)
    assert not is_regressor(clone(selector).set_params(refit=False))
    assert np.isfinite(cross_val_score(selector, X, y, cv=3, scoring="r2")).all()


def test_outlier_selector_exposes_refitted_sample_scores():
    X, _ = load_iris(return_X_y=True)
    selector = GAFeatureSelectionCV(
        IsolationForest(n_estimators=5, random_state=31, n_jobs=1),
        cv=2,
        population_size=4,
        generations=0,
        random_state=31,
        n_jobs=1,
        verbose=False,
    ).fit(X)
    assert is_outlier_detector(selector)
    assert not is_outlier_detector(clone(selector).set_params(refit=False))
    expected = selector.best_estimator_.score_samples(selector.transform(X))
    np.testing.assert_array_equal(selector.score_samples(X), expected)
    assert selector.score(X) == pytest.approx(expected.mean())
    weights = np.arange(1, len(X) + 1)
    assert selector.score(X, sample_weight=weights) == pytest.approx(
        np.average(expected, weights=weights)
    )


@pytest.mark.parametrize(
    "scoring, refit, objective_metric",
    [
        (None, True, None),
        ("accuracy", True, None),
        ({"acc": "accuracy", "f1": "f1_macro"}, "acc", None),
        ({"acc": "accuracy", "f1": "f1_macro"}, lambda results: 0, "acc"),
    ],
)
def test_feature_selector_weighted_score_uses_the_refitted_objective(
    scoring, refit, objective_metric
):
    X, y = load_iris(return_X_y=True)
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(max_depth=1, random_state=31),
        cv=2,
        population_size=4,
        generations=0,
        random_state=31,
        n_jobs=1,
        verbose=False,
        scoring=scoring,
        refit=refit,
        objective_metric=objective_metric,
    ).fit(X, y)
    weights = np.arange(1, len(y) + 1)
    expected = accuracy_score(y, selector.predict(X), sample_weight=weights)
    assert expected != accuracy_score(y, selector.predict(X))
    assert selector.score(X, y, sample_weight=weights) == pytest.approx(expected)


@pytest.mark.parametrize("route", [False, True])
def test_feature_selector_scores_weighted_pipeline_output(route):
    from sklearn import config_context
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    weights = np.arange(1, len(y) + 1)
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(max_depth=1, random_state=31),
        cv=2,
        population_size=4,
        generations=0,
        random_state=31,
        n_jobs=1,
        verbose=False,
    )
    with config_context(enable_metadata_routing=route):
        if route:
            selector.set_score_request(sample_weight=True)
        pipeline = make_pipeline(StandardScaler(), selector).fit(X, y)
        expected = accuracy_score(y, pipeline.predict(X), sample_weight=weights)
        assert pipeline.score(X, y, sample_weight=weights) == pytest.approx(expected)


@pytest.mark.parametrize("alias", ["sample_weight", "weights"])
def test_external_validation_routes_only_held_out_score_weights(alias):
    from sklearn import config_context

    class UnweightedFitTree(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            assert sample_weight is None
            return super().fit(X, y, check_input=check_input)

    X, y = load_iris(return_X_y=True)
    weights = np.arange(1, len(y) + 1)
    selector = GAFeatureSelectionCV(
        UnweightedFitTree(max_depth=1, random_state=31),
        cv=2,
        population_size=4,
        generations=0,
        random_state=31,
        n_jobs=1,
        verbose=False,
    )
    with config_context(enable_metadata_routing=True):
        selector.estimator.set_fit_request(sample_weight=False).set_score_request(
            sample_weight=False
        )
        selector.set_score_request(sample_weight=True if alias == "sample_weight" else alias)
        scores = cross_validate(
            selector,
            X,
            y,
            cv=3,
            params={alias: weights},
            return_indices=True,
            return_estimator=True,
            error_score="raise",
        )
    for fitted, indices, value in zip(
        scores["estimator"], scores["indices"]["test"], scores["test_score"], strict=True
    ):
        expected = accuracy_score(
            y[indices], fitted.predict(X[indices]), sample_weight=weights[indices]
        )
        assert value == pytest.approx(expected)


@pytest.mark.parametrize("backend", ["deap", "scipy", "nevergrad", "cmaes", "pymoo"])
@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("group_request", ["splitter", "explicit", "alias"])
def test_pipeline_routes_fit_metadata_to_a_feature_selector(backend, refit, group_request):
    from sklearn import config_context
    from sklearn.pipeline import make_pipeline

    if backend != "deap":
        pytest.importorskip(backend)
    rows = np.arange(36)
    X = np.column_stack([rows + offset * len(rows) for offset in range(4)])
    y = rows % 2
    groups = rows // 6
    training = rows + 1
    evaluation = rows[::-1] + 100
    observed = []

    class WeightCheckingTree(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None, check_input=True):
            np.testing.assert_array_equal(sample_weight, training[X[:, 0] % len(rows)])
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

    def weighted_accuracy(y_true, y_pred, sample_weight=None):
        observed.append(np.asarray(sample_weight).tolist())
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    with config_context(enable_metadata_routing=True):
        estimator = WeightCheckingTree(random_state=31).set_fit_request(
            sample_weight="training_weight"
        )
        scorer = make_scorer(weighted_accuracy).set_score_request(sample_weight="evaluation_weight")
        selector = GAFeatureSelectionCV(
            estimator,
            optimizer_backend=backend,
            scoring={"acc": scorer, "secondary": "balanced_accuracy"},
            refit="acc" if refit else False,
            objective_metric=None if refit else "acc",
            cv=GroupKFold(2),
            population_size=6,
            generations=0,
            max_features=2,
            random_state=31,
            n_jobs=1,
            verbose=False,
            error_score="raise",
        )
        group_name = "groups"
        if group_request == "explicit":
            selector.set_fit_request(groups=True)
        elif group_request == "alias":
            group_name = "selection_groups"
            selector.set_fit_request(groups=group_name)
        pipeline = (
            make_pipeline(selector)
            if refit
            else make_pipeline(selector, DecisionTreeClassifier(random_state=31))
        )
        pipeline.fit(
            X, y, training_weight=training, evaluation_weight=evaluation, **{group_name: groups}
        )
    fitted = pipeline.steps[0][1]
    np.testing.assert_array_equal(fitted.groups_, groups)
    folds = list(GroupKFold(2).split(X, y, groups))
    assert (
        observed
        == [evaluation[test].tolist() for _, test in folds] * fitted.fit_counts_["search_cv_calls"]
    )
    assert fitted.fit_counts_["refit_fits"] == int(refit)
    assert len(pipeline.predict(X)) == len(y)


@pytest.mark.parametrize("backend", ["deap", "scipy", "nevergrad", "cmaes", "pymoo"])
@pytest.mark.parametrize("group_request", ["automatic", "alias"])
def test_nested_pipeline_routes_groups_to_feature_finalists_only(backend, group_request):
    from sklearn import config_context
    from sklearn.pipeline import make_pipeline

    if backend != "deap":
        pytest.importorskip(backend)
    rows = np.arange(48)
    X = np.column_stack([rows + offset * len(rows) for offset in range(4)])
    y = rows % 2
    outer_groups = rows // 6
    inner_groups = outer_groups if group_request == "automatic" else rows // 3
    observed = []

    class UngroupedSearchFolds:
        def split(self, X, y):
            return StratifiedKFold(2).split(X, y)

    class FinalistFolds(GroupKFold):
        def split(self, X, y=None, groups=None):
            observed.append(np.asarray(groups).copy())
            for train, test in super().split(X, y, groups):
                assert not np.intersect1d(groups[train], groups[test]).size
                yield train, test

    with config_context(enable_metadata_routing=True):
        selector = GAFeatureSelectionCV(
            DecisionTreeClassifier(random_state=17),
            cv=UngroupedSearchFolds(),
            final_selection=True,
            final_selection_cv=FinalistFolds(2),
            final_selection_top_k=2,
            optimizer_backend=backend,
            population_size=6,
            generations=0,
            max_features=2,
            n_jobs=1,
            random_state=17,
            verbose=False,
            error_score="raise",
        )
        params = {"groups": outer_groups}
        if group_request == "alias":
            selector.set_fit_request(groups="finalist_groups")
            params["finalist_groups"] = inner_groups
        pipeline = make_pipeline(selector, DecisionTreeClassifier(random_state=17))
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=GroupKFold(3),
            params=params,
            return_indices=True,
            return_estimator=True,
            error_score="raise",
        )

    trains = result["indices"]["train"]
    assert len(observed) == len(trains) == 3
    for train, seen, fitted in zip(trains, observed, result["estimator"], strict=True):
        actual = fitted.steps[0][1]
        np.testing.assert_array_equal(seen, inner_groups[train])
        np.testing.assert_array_equal(actual.groups_, inner_groups[train])
        assert actual.fit_counts_["search_fold_fits"] == 2 * actual.fit_counts_["search_cv_calls"]
        assert actual.fit_counts_["finalist_fold_fits"] == 4


@pytest.mark.parametrize("inactive", ["refit", "final_selection", "explicit"])
def test_pipeline_requests_no_groups_for_unused_feature_finalists(inactive):
    from sklearn import config_context

    with config_context(enable_metadata_routing=True):
        selector = GAFeatureSelectionCV(
            DecisionTreeClassifier(),
            cv=2,
            refit=inactive != "refit",
            final_selection=inactive != "final_selection",
            final_selection_cv=GroupKFold(2),
        )
        if inactive == "explicit":
            selector.set_fit_request(groups=False)
        assert not selector.get_metadata_routing().consumes("fit", ["groups"])


def test_external_validation_routes_feature_selector_fit_and_score_metadata():
    from sklearn import config_context

    X, y = load_iris(return_X_y=True)
    training = np.arange(1, len(y) + 1)
    evaluation = training[::-1].copy()
    outer = np.linspace(0.25, 2.0, len(y))
    with config_context(enable_metadata_routing=True):
        estimator = DecisionTreeClassifier(max_depth=1, random_state=31).set_fit_request(
            sample_weight="training_weight"
        )
        scorer = make_scorer(accuracy_score).set_score_request(sample_weight="evaluation_weight")
        selector = GAFeatureSelectionCV(
            estimator,
            scoring=scorer,
            cv=2,
            population_size=4,
            generations=0,
            random_state=31,
            n_jobs=1,
            verbose=False,
        ).set_score_request(sample_weight="outer_weight")
        result = cross_validate(
            selector,
            X,
            y,
            cv=3,
            params={
                "training_weight": training,
                "evaluation_weight": evaluation,
                "outer_weight": outer,
            },
            return_indices=True,
            return_estimator=True,
            error_score="raise",
        )
    for train, test, fitted, value in zip(
        result["indices"]["train"],
        result["indices"]["test"],
        result["estimator"],
        result["test_score"],
        strict=True,
    ):
        np.testing.assert_array_equal(fitted._fit_params["training_weight"], training[train])
        np.testing.assert_array_equal(fitted._fit_params["evaluation_weight"], evaluation[train])
        expected = accuracy_score(y[test], fitted.predict(X[test]), sample_weight=outer[test])
        assert value == pytest.approx(expected)


def test_feature_selection_preserves_strict_finite_validation():
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="Input X contains NaN"):
        GAFeatureSelectionCV(LogisticRegression()).fit(X, y)

    X[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        GAFeatureSelectionCV(DecisionTreeClassifier()).fit(X, y)

    X[0, 0] = 0.0
    y = y.astype(float)
    y[0] = np.nan
    with pytest.raises(ValueError, match="Input y contains NaN"):
        GAFeatureSelectionCV(DecisionTreeClassifier()).fit(X, y)


def test_default_n_jobs_is_none():
    estimator = GAFeatureSelectionCV(
        SGDClassifier(loss="modified_huber", fit_intercept=True),
        generations=1,
        population_size=2,
        verbose=False,
    )

    assert estimator.n_jobs is None
    assert estimator.get_params()["n_jobs"] is None
    assert estimator.parallel_backend == "auto"
    assert estimator.get_params()["parallel_backend"] == "auto"
    assert estimator.population_initializer == "smart"
    assert estimator.get_params()["population_initializer"] == "smart"


def test_feature_selection_accepts_grouped_config_objects():
    evolution_config = EvolutionConfig(population_size=5, generations=2, tournament_size=2)
    population_config = PopulationConfig(initializer="smart")
    runtime_config = RuntimeConfig(n_jobs=1, parallel_backend="auto", verbose=False)
    optimization_config = OptimizationConfig(
        local_search=True,
        local_search_top_k=2,
        diversity_control=True,
        adaptive_selection=True,
        offspring_diversity_retries=2,
        fitness_sharing=True,
    )

    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        max_features=5,
        evolution_config=evolution_config,
        population_config=population_config,
        runtime_config=runtime_config,
        optimization_config=optimization_config,
    )

    assert estimator.population_size == 5
    assert estimator.generations == 2
    assert estimator.tournament_size == 2
    assert estimator.population_initializer == "smart"
    assert estimator.n_jobs == 1
    assert estimator.verbose is False
    assert estimator.local_search is True
    assert estimator.local_search_top_k == 2
    assert estimator.diversity_control is True
    assert estimator.adaptive_selection is True
    assert estimator.offspring_diversity_retries == 2
    assert estimator.fitness_sharing is True
    assert estimator.get_params()["evolution_config"] is evolution_config
    assert estimator.get_params()["runtime_config"] is runtime_config


def test_feature_selection_grouped_config_is_sklearn_clone_compatible():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        evolution_config=EvolutionConfig(population_size=4, generations=1),
        population_config=PopulationConfig(initializer="smart"),
        runtime_config=RuntimeConfig(verbose=False),
    )

    cloned = clone(estimator)

    assert cloned.population_size == 4
    assert cloned.generations == 1
    assert cloned.population_initializer == "smart"
    assert cloned.verbose is False
    assert "runtime_config" in cloned.get_params()


def test_feature_selection_rejects_invalid_error_score():
    with pytest.raises(ValueError) as excinfo:
        GAFeatureSelectionCV(
            DecisionTreeClassifier(),
            error_score="warn",
        )

    assert str(excinfo.value) == "error_score must be numeric or 'raise', got 'warn' instead"


@pytest.mark.parametrize("error_score", ["raise", np.nan, 0.0])
def test_feature_selection_accepts_valid_error_score(error_score):
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(),
        error_score=error_score,
    )

    if isinstance(error_score, float) and np.isnan(error_score):
        assert np.isnan(estimator.error_score)
    else:
        assert estimator.error_score == error_score


@pytest.mark.parametrize("fitness_sharing", [False, True])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
@pytest.mark.parametrize("criteria", ["max", "min"])
def test_nonfinite_scoring_mask_cannot_displace_valid_feature_elite(
    criteria, nonfinite, fitness_sharing
):
    iris = load_iris()
    direction = 1 if criteria == "max" else -1
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=0),
        scoring=lambda estimator, X, y: (
            direction * nonfinite if X.shape[1] == 1 else direction * X.shape[1]
        ),
        cv=2,
        population_size=6,
        generations=1,
        criteria=criteria,
        fitness_sharing=fitness_sharing,
        random_state=3,
        verbose=False,
        n_jobs=1,
    ).fit(iris.data[:60], iris.target[:60])

    assert selector.support_.sum() == iris.data.shape[1]
    assert selector.history["fitness_best"][-1] == direction * iris.data.shape[1]
    invalid_indices = [
        index for index, mask in enumerate(selector.cv_results_["features"]) if sum(mask) == 1
    ]
    assert invalid_indices
    for index in invalid_indices:
        for score in (
            selector.cv_results_["mean_test_score"][index],
            *(selector.cv_results_[f"split{split}_test_score"][index] for split in range(2)),
        ):
            if np.isnan(nonfinite):
                assert np.isnan(score)
            else:
                assert score == direction * nonfinite


@pytest.mark.parametrize("kind", ["hpo", "features"])
@pytest.mark.parametrize("criteria", ["max", "min"])
def test_partially_undefined_candidate_cannot_win_final_selection(kind, criteria):
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Categorical, Integer

    values = np.arange(60).reshape(30, 2)
    labels = np.arange(30) % 2
    indices = np.arange(30)
    cv = [(indices[~np.isin(indices, test)], test) for test in np.split(indices, 3)]
    direction = 1 if criteria == "max" else -1

    def scorer(estimator, X, y):
        size = estimator.max_depth if kind == "hpo" else X.shape[1]
        if size == 1:
            return np.nan if X[0, 0] < 10 else direction * 100.0
        return direction * size

    common = dict(
        cv=cv,
        scoring=scorer,
        population_size=6,
        generations=1,
        criteria=criteria,
        random_state=3,
        n_jobs=1,
        verbose=False,
        optimization_config=OptimizationConfig(final_selection=True, final_selection_top_k=1),
    )
    model = DecisionTreeClassifier(random_state=0)
    search = (
        GASearchCV(
            model,
            param_grid={"max_depth": Categorical([1, 2, 3]), "min_samples_leaf": Integer(1, 3)},
            **common,
        )
        if kind == "hpo"
        else GAFeatureSelectionCV(model, **common)
    ).fit(values, labels)

    assert np.isfinite(search.best_score_)
    if kind == "hpo":
        assert search.best_params_["max_depth"] == 3
    else:
        assert search.support_.sum() == 2
    assert any(np.isnan(score) for score in search.cv_results_["mean_test_score"])


def _checkpoint_nonfinite_score(estimator, X, y, invalid):
    objective = estimator.max_depth if estimator.max_depth is not None else X.shape[1]
    return invalid if objective == 1 else float(objective)


@pytest.mark.parametrize("kind", ["hpo", "features"])
@pytest.mark.parametrize("criteria", ["max", "min"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_resume_normalizes_legacy_nonfinite_cached_fitness(kind, criteria, nonfinite, tmp_path):
    import pickle

    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Categorical, Integer
    from ..callbacks import ModelCheckpoint

    values, labels = load_iris(return_X_y=True)
    path = str(tmp_path / "legacy_nonfinite_fitness.pkl")
    invalid_score = nonfinite if np.isnan(nonfinite) or criteria == "max" else -nonfinite
    common = dict(
        cv=2,
        scoring=partial(_checkpoint_nonfinite_score, invalid=invalid_score),
        population_size=6,
        generations=1,
        criteria=criteria,
        random_state=3,
        n_jobs=1,
        verbose=False,
    )

    def build():
        model = DecisionTreeClassifier(random_state=0)
        if kind == "hpo":
            return GASearchCV(
                model,
                param_grid={
                    "max_depth": Categorical([1, 2, 3]),
                    "min_samples_leaf": Integer(1, 10),
                },
                **common,
            )
        return GAFeatureSelectionCV(model, **common)

    original = build().fit(values[:60], labels[:60], callbacks=ModelCheckpoint(path))
    with open(path, "rb") as file:
        checkpoint = pickle.load(file)
    cached = checkpoint["runtime_state"]["fitness_cache"]
    record_key = "current_generation_params" if kind == "hpo" else "current_generation_features"
    invalid = [item for item in cached.values() if not np.isfinite(item[record_key]["score"])]
    assert invalid
    assert all(not np.isnan(item["fitness"][0]) for item in invalid)
    for item in invalid:
        fitness = item["fitness"]
        item["fitness"] = (
            (invalid_score, *fitness[1:]) if kind == "hpo" else [invalid_score, *fitness[1:]]
        )
    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)

    common["criteria"] = "min" if criteria == "max" else "max"
    resumed = build().set_params(criteria=criteria)
    resumed.fit(values[:60], labels[:60], callbacks=ModelCheckpoint(path))
    assert resumed.fit_stats_["cache_hits"] > original.fit_stats_["cache_hits"]
    assert np.isfinite(resumed.history["fitness_best"][-1])
    assert any(
        (np.isnan(score) if np.isnan(nonfinite) else score == invalid_score)
        for score in resumed.cv_results_["mean_test_score"]
    )
    if kind == "hpo":
        assert resumed.best_params_["max_depth"] == (3 if criteria == "max" else 2)
    else:
        assert resumed.support_.sum() == (values.shape[1] if criteria == "max" else 2)
        index = resumed.best_index_
        np.testing.assert_equal(resumed.cv_results_["features"][index], resumed.get_support())
        assert resumed.best_score_ == resumed.cv_results_["mean_test_score"][index]


def test_smart_population_initializer_creates_diverse_feature_masks():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        population_size=5,
        generations=1,
        max_features=3,
        verbose=False,
    )
    estimator.n_features = 6
    estimator.features_proportion = estimator.max_features / estimator.n_features

    estimator._register()
    population = [list(individual) for individual in estimator._pop]
    selected_counts = [sum(individual) for individual in population]

    assert len(population) == estimator.population_size
    assert len({tuple(individual) for individual in population}) == len(population)
    assert max(selected_counts) <= estimator.max_features
    assert min(selected_counts) >= 1
    assert len(set(selected_counts)) > 1


def test_optimizer_telemetry_is_recorded_for_feature_selection():
    generations = 2
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring=lambda estimator, X, y: 1.0,
        population_size=4,
        generations=generations,
        algorithm="eaSimple",
        verbose=False,
        n_jobs=1,
    )

    estimator.fit(X_train, y_train)

    telemetry_fields = [
        "population_size",
        "fitness_best",
        "unique_individuals",
        "unique_individual_ratio",
        "genotype_diversity",
        "fitness_improvement",
        "fitness_improved",
        "stagnation_generations",
        "best_generation",
    ]

    for field in telemetry_fields:
        assert field in estimator.history
        assert field in estimator[0]
        assert len(estimator.history[field]) == generations + 1

    assert estimator.history["population_size"][-1] == estimator.population_size
    assert 0 <= estimator.history["unique_individual_ratio"][-1] <= 1
    assert 0 <= estimator.history["genotype_diversity"][-1] <= 1
    assert estimator.history["best_generation"][-1] == 0
    assert estimator.history["stagnation_generations"][-1] == generations


class _ScalarAssignmentMask(list):
    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise TypeError("This mask only supports single-element assignment")
        super().__setitem__(index, value)


@pytest.mark.parametrize("mask_type", [list, partial(array, "i"), _ScalarAssignmentMask])
@pytest.mark.parametrize("values", [[0, -3, 2, 0], [0, 0, 0, 0]])
def test_feature_masks_are_repaired_before_evaluation(mask_type, values):
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        max_features=1,
        population_size=4,
        generations=0,
        random_state=42,
        n_jobs=1,
        verbose=False,
    ).fit(data.data, data.target)

    individual = mask_type(values)
    (fitness,) = estimator.evaluate_population([individual])

    assert sum(individual) == 1
    assert fitness[1] == 1
    assert 0 <= fitness[0] <= 1
    assert set(individual).issubset({0, 1})
    assert all(type(value) is int for value in individual)
    if any(values):
        assert individual[0] == individual[3] == 0


def test_feature_masks_are_repaired_when_max_features_is_missing():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(),
        cv=3,
        scoring="accuracy",
        verbose=False,
    )
    if hasattr(estimator, "max_features"):
        delattr(estimator, "max_features")

    individual = [1, 1, 0]
    repaired = estimator._repair_individual(individual)

    assert repaired is individual
    assert sum(individual) >= 1
    assert set(individual).issubset({0, 1})


def test_feature_selection_genetic_operations_respect_max_features():
    estimator = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=42),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=1,
        max_features=2,
        verbose=False,
    )
    estimator.n_features = 6
    estimator.features_proportion = estimator.max_features / estimator.n_features

    estimator._register()

    individual_type = type(estimator._pop[0])
    first = individual_type([1, 1, 1, 0, 0, 0])
    second = individual_type([0, 0, 0, 1, 1, 1])

    offspring = estimator.mate(first, second)
    mutated = estimator.mutate(individual_type([1, 1, 1, 1, 1, 1]))

    for individual in (*offspring, *mutated, estimator.toolbox.individual()):
        assert 1 <= sum(individual) <= estimator.max_features
        assert set(individual).issubset({0, 1})


def test_feature_toolbox_clone_preserves_mask_and_metadata_independence(monkeypatch):
    selector = GAFeatureSelectionCV(DecisionTreeClassifier(), population_size=4)
    selector.n_features = 4
    selector.features_proportion = 0.5
    selector._register()
    parent = selector.toolbox.individual()
    parent.fitness.values = (0.7, sum(parent))
    canonical = selector.toolbox.clone(parent)
    assert canonical == parent and canonical.fitness.values == parent.fitness.values
    canonical[0] = 1 - canonical[0]
    del canonical.fitness.values
    assert canonical != parent and parent.fitness.valid
    parent.notes = {"owner": parent, "scores": [0.7]}
    parent.same_notes = parent.notes
    child = selector.toolbox.clone(parent)
    assert child == parent and type(child) is type(parent)
    assert child.fitness.values == parent.fitness.values
    assert child.notes is child.same_notes and child.notes["owner"] is child
    child[0] = 1 - child[0]
    del child.fitness.values
    child.notes["scores"].append(0.8)
    assert child != parent and parent.fitness.valid
    assert parent.notes["scores"] == [0.7]

    # DEAP's generic clone also supports memoized copies and mutable payloads.
    sentinel = []
    assert selector.toolbox.clone(parent, {id(parent): sentinel}) is sentinel
    parent[0] = parent.notes["scores"]
    unusual = selector.toolbox.clone(parent)
    assert unusual[0] is unusual.notes["scores"] and unusual[0] is not parent[0]
    canonical.__deepcopy__ = lambda memo: sentinel
    assert selector.toolbox.clone(canonical) is sentinel
    del canonical.__deepcopy__
    monkeypatch.setattr(type(canonical), "__deepcopy__", lambda self, memo: sentinel, raising=False)
    assert selector.toolbox.clone(canonical) is sentinel


def test_expected_ga_results():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "split0_test_score" in cv_result_keys
    assert "split1_test_score" in cv_result_keys
    assert "split2_test_score" in cv_result_keys
    assert "split0_train_score" in cv_result_keys
    assert "split1_train_score" in cv_result_keys
    assert "split2_train_score" in cv_result_keys
    assert "mean_test_score" in cv_result_keys
    assert "std_test_score" in cv_result_keys
    assert "rank_test_score" in cv_result_keys
    assert "mean_train_score" in cv_result_keys
    assert "std_train_score" in cv_result_keys
    assert "rank_train_score" in cv_result_keys
    assert "std_fit_time" in cv_result_keys
    assert "mean_score_time" in cv_result_keys
    assert "rank_n_features" in cv_result_keys
    assert "features" in cv_result_keys


@pytest.mark.parametrize(
    "algorithm, callback",
    [
        ("eaSimple", ThresholdStopping(threshold=0.01)),
        ("eaMuPlusLambda", ThresholdStopping(threshold=0.01)),
        ("eaMuCommaLambda", ThresholdStopping(threshold=0.01)),
        ("eaSimple", TimerStopping(total_seconds=0.5)),
        ("eaMuPlusLambda", TimerStopping(total_seconds=2)),
        ("eaMuCommaLambda", TimerStopping(total_seconds=5)),
        ("eaSimple", ConsecutiveStopping(generations=3, metric="fitness")),
        ("eaMuPlusLambda", ConsecutiveStopping(generations=3, metric="fitness")),
        ("eaMuCommaLambda", ConsecutiveStopping(generations=3, metric="fitness")),
        ("eaSimple", DeltaThreshold(threshold=0.001, metric="fitness")),
        ("eaMuPlusLambda", DeltaThreshold(threshold=0.001, metric="fitness")),
        ("eaMuCommaLambda", DeltaThreshold(threshold=0.001, metric="fitness")),
        ("eaSimple", ProgressBar()),
        (
            "eaMuPlusLambda",
            ProgressBar(**{"desc": "my_custom_desc", "mininterval": 0.5}),
        ),
        ("eaMuCommaLambda", ProgressBar()),
        (
            "eaSimple",
            [
                ThresholdStopping(threshold=0.01),
                ConsecutiveStopping(generations=5, metric="fitness"),
                DeltaThreshold(threshold=0.001, metric="fitness"),
            ],
        ),
        (
            "eaMuPlusLambda",
            [
                ThresholdStopping(threshold=0.01),
                ConsecutiveStopping(generations=5, metric="fitness"),
                DeltaThreshold(threshold=0.001, metric="fitness"),
            ],
        ),
        (
            "eaMuCommaLambda",
            [
                ThresholdStopping(threshold=0.01),
                ConsecutiveStopping(generations=5, metric="fitness"),
                DeltaThreshold(threshold=0.001, metric="fitness"),
            ],
        ),
    ],
)
def test_expected_algorithms_callbacks(algorithm, callback):
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=2,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=True,
        algorithm=algorithm,
        n_jobs=-1,
    )

    evolved_estimator.fit(X_train, y_train, callbacks=callback)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
    assert len(evolved_estimator) <= generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]


def test_negative_criteria():
    data_boston = load_diabetes()

    y_diabetes = data_boston["target"]
    X_diabetes = data_boston["data"]

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_diabetes, y_diabetes, test_size=0.33, random_state=42
    )

    clf = DecisionTreeRegressor()
    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="neg_mean_squared_error",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.9,
        mutation_probability=0.05,
        criteria="min",
        n_jobs=-1,
    )

    evolved_estimator.fit(X_train_b, y_train_b)

    assert check_is_fitted(evolved_estimator) is None
    assert len(evolved_estimator.predict(X_test_b)) == len(X_test_b)


def test_wrong_criteria():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    with pytest.raises(Exception) as excinfo:
        evolved_estimator = GAFeatureSelectionCV(
            clf,
            cv=3,
            scoring="accuracy",
            population_size=5,
            generations=generations,
            tournament_size=3,
            elitism=False,
            verbose=False,
            criteria="maximization",
        )
    assert str(excinfo.value) == "Criteria must be one of ['max', 'min'], got maximization instead"


def test_wrong_estimator():
    clf = KMeans()
    generations = 8
    with pytest.raises(Exception) as excinfo:
        evolved_estimator = GAFeatureSelectionCV(
            clf,
            cv=3,
            scoring="accuracy",
            population_size=5,
            generations=generations,
            tournament_size=3,
            elitism=False,
            verbose=False,
            criteria="maximization",
        )
    assert (
        str(excinfo.value)
        == "KMeans() is not a valid Sklearn classifier, regressor, or outlier detector"
    )


def test_wrong_get_item():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 8
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=False,
        verbose=False,
        criteria="max",
    )
    with pytest.raises(Exception) as excinfo:
        value = evolved_estimator[0]
    assert (
        str(excinfo.value)
        == "This GAFeatureSelectionCV instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )


def test_iterator():
    clf = DecisionTreeClassifier()
    generations = 4
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=3,
        generations=generations,
        tournament_size=3,
        elitism=True,
        verbose=False,
        n_jobs=-1,
    )
    evolved_estimator.fit(X_train, y_train)

    i = iter(evolved_estimator)
    assert next(i) == evolved_estimator[0]
    assert next(i) == evolved_estimator[1]


def test_wrong_algorithm():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=5,
        generations=generations,
        tournament_size=3,
        elitism=False,
        verbose=False,
        criteria="max",
        algorithm="genetic",
    )
    with pytest.raises(Exception) as excinfo:
        evolved_estimator.fit(X_train, y_train)
    assert (
        str(excinfo.value)
        == "The algorithm genetic is not supported, please select one from ['eaSimple', 'eaMuPlusLambda', 'eaMuCommaLambda', 'eaDifferentialEvolution']"
    )


def test_expected_ga_max_features():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    noise_train = np.random.uniform(0, 16, size=(X_train.shape[0], 16))
    noise_test = np.random.uniform(0, 16, size=(X_test.shape[0], 16))
    X_noise_train = np.hstack((X_train, noise_train))
    X_noise_test = np.hstack((X_test, noise_test))
    generations = 8
    max_features = 10
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=10,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        max_features=max_features,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
    )

    evolved_estimator.fit(X_noise_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X_noise_train.shape[1]
    assert sum(features) <= max_features
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_noise_test)) == len(X_noise_test)
    assert evolved_estimator.score(X_noise_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_noise_test)) == len(X_noise_test)
    assert len(evolved_estimator.predict_proba(X_noise_test)) == len(X_noise_test)
    assert len(evolved_estimator.predict_log_proba(X_noise_test)) == len(X_noise_test)
    assert evolved_estimator.score(X_noise_test, y_test) == accuracy_score(
        evolved_estimator.predict(X_noise_test), y_test
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "split0_test_score" in cv_result_keys
    assert "split1_test_score" in cv_result_keys
    assert "split2_test_score" in cv_result_keys
    assert "split0_train_score" in cv_result_keys
    assert "split1_train_score" in cv_result_keys
    assert "split2_train_score" in cv_result_keys
    assert "mean_test_score" in cv_result_keys
    assert "std_test_score" in cv_result_keys
    assert "rank_test_score" in cv_result_keys
    assert "mean_train_score" in cv_result_keys
    assert "std_train_score" in cv_result_keys
    assert "rank_train_score" in cv_result_keys
    assert "std_fit_time" in cv_result_keys
    assert "mean_score_time" in cv_result_keys
    assert "rank_n_features" in cv_result_keys
    assert "features" in cv_result_keys


def test_expected_ga_multimetric():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }

    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring=scoring,
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
        refit="accuracy",
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "std_fit_time" in cv_result_keys
    assert "features" in cv_result_keys

    for metric in scoring.keys():
        assert f"split0_test_{metric}" in cv_result_keys
        assert f"split1_test_{metric}" in cv_result_keys
        assert f"split2_test_{metric}" in cv_result_keys
        assert f"split0_train_{metric}" in cv_result_keys
        assert f"split1_train_{metric}" in cv_result_keys
        assert f"split2_train_{metric}" in cv_result_keys
        assert f"mean_test_{metric}" in cv_result_keys
        assert f"std_test_{metric}" in cv_result_keys
        assert f"rank_test_{metric}" in cv_result_keys
        assert f"mean_train_{metric}" in cv_result_keys
        assert f"std_train_{metric}" in cv_result_keys
        assert f"rank_train_{metric}" in cv_result_keys


@pytest.mark.parametrize(
    ("scoring", "refit"),
    [
        ("f1", True),
        (make_scorer(f1_score), True),
        ({"accuracy": "accuracy", "f1": "f1"}, "f1"),
    ],
)
def test_feature_selection_score_uses_configured_scorer(scoring, refit):
    X = np.arange(20).reshape(-1, 1)
    y = np.array([0] * 16 + [1] * 4)
    selector = GAFeatureSelectionCV(
        DummyClassifier(strategy="most_frequent"),
        cv=2,
        scoring=scoring,
        refit=refit,
        population_size=4,
        generations=1,
        n_jobs=1,
        random_state=42,
    ).fit(X, y)

    assert selector.score(X, y) == 0.0
    assert selector.best_estimator_.score(selector.transform(X), y) == 0.8


def test_feature_selection_score_supports_a_classifier_without_score():
    class PredictOnlyClassifier(DummyClassifier):
        @property
        def score(self):
            raise AttributeError("score")

    X, y = load_iris(return_X_y=True)
    selector = GAFeatureSelectionCV(
        PredictOnlyClassifier(strategy="most_frequent"),
        cv=3,
        scoring="accuracy",
        population_size=3,
        generations=0,
        n_jobs=1,
        random_state=4,
        verbose=False,
    ).fit(X, y)

    assert selector.score(X, y) == accuracy_score(y, selector.predict(X))


def test_expected_ga_callable_score():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    scoring = make_scorer(accuracy_score)
    generations = 6
    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring=scoring,
        population_size=6,
        generations=generations,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
        refit="accuracy",
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "split0_test_score" in cv_result_keys
    assert "split1_test_score" in cv_result_keys
    assert "split2_test_score" in cv_result_keys
    assert "split0_train_score" in cv_result_keys
    assert "split1_train_score" in cv_result_keys
    assert "split2_train_score" in cv_result_keys
    assert "mean_test_score" in cv_result_keys
    assert "std_test_score" in cv_result_keys
    assert "rank_test_score" in cv_result_keys
    assert "mean_train_score" in cv_result_keys
    assert "std_train_score" in cv_result_keys
    assert "rank_train_score" in cv_result_keys
    assert "std_fit_time" in cv_result_keys
    assert "mean_score_time" in cv_result_keys
    assert "rank_n_features" in cv_result_keys
    assert "features" in cv_result_keys


def test_expected_ga_schedulers():
    clf = SGDClassifier(loss="modified_huber", fit_intercept=True)
    generations = 6
    mutation_scheduler = ExponentialAdapter(initial_value=0.6, adaptive_rate=0.01, end_value=0.2)
    crossover_scheduler = InverseAdapter(initial_value=0.4, adaptive_rate=0.01, end_value=0.3)

    evolved_estimator = GAFeatureSelectionCV(
        clf,
        cv=3,
        scoring="accuracy",
        population_size=6,
        generations=generations,
        mutation_probability=mutation_scheduler,
        crossover_probability=crossover_scheduler,
        tournament_size=3,
        elitism=False,
        keep_top_k=4,
        verbose=False,
        algorithm="eaSimple",
        n_jobs=-1,
        return_train_score=True,
        refit="accuracy",
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.support_

    assert check_is_fitted(evolved_estimator) is None
    assert features.shape[0] == X.shape[1]
    assert len(evolved_estimator) == generations + 1  # +1 random initial population
    assert len(evolved_estimator.predict(X_test)) == len(X_test)
    assert evolved_estimator.score(X_train, y_train) >= 0
    assert len(evolved_estimator.decision_function(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_proba(X_test)) == len(X_test)
    assert len(evolved_estimator.predict_log_proba(X_test)) == len(X_test)
    assert evolved_estimator.score(X_test, y_test) == accuracy_score(
        y_test, evolved_estimator.predict(X_test)
    )
    assert bool(evolved_estimator.get_params())
    assert len(evolved_estimator.hof) == evolved_estimator.keep_top_k
    assert "gen" in evolved_estimator[0]
    assert "fitness_max" in evolved_estimator[0]
    assert "fitness" in evolved_estimator[0]
    assert "fitness_std" in evolved_estimator[0]
    assert "fitness_min" in evolved_estimator[0]

    cv_results_ = evolved_estimator.cv_results_
    cv_result_keys = set(cv_results_.keys())

    assert "split0_test_score" in cv_result_keys
    assert "split1_test_score" in cv_result_keys
    assert "split2_test_score" in cv_result_keys
    assert "split0_train_score" in cv_result_keys
    assert "split1_train_score" in cv_result_keys
    assert "split2_train_score" in cv_result_keys
    assert "mean_test_score" in cv_result_keys
    assert "std_test_score" in cv_result_keys
    assert "rank_test_score" in cv_result_keys
    assert "mean_train_score" in cv_result_keys
    assert "std_train_score" in cv_result_keys
    assert "rank_train_score" in cv_result_keys
    assert "std_fit_time" in cv_result_keys
    assert "mean_score_time" in cv_result_keys
    assert "rank_n_features" in cv_result_keys
    assert "features" in cv_result_keys

    assert crossover_scheduler.current_value + mutation_scheduler.current_value <= 1


def test_feature_selection_fit_with_groups_supports_group_kfold():
    """#338: GAFeatureSelectionCV.fit(..., groups=...) enables group-aware CV.

    Every materialized split must keep train and test groups disjoint, and the
    selector must still produce a valid support mask.
    """
    from sklearn.model_selection import GroupKFold

    groups = np.arange(X_train.shape[0]) % 3
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=0),
        cv=GroupKFold(n_splits=3),
        scoring="accuracy",
        population_size=4,
        generations=2,
        verbose=False,
    )

    selector.fit(X_train, y_train, groups=groups)

    assert selector.support_.shape[0] == X_train.shape[1]
    assert len(selector._cv_splits) == 3
    for train_idx, test_idx in selector._cv_splits:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_feature_selection_forwards_sample_weight_and_slices_only_features():
    """#339: GAFeatureSelectionCV forwards sample_weight while only X is
    feature-sliced.

    Candidate fits see a feature-sliced X but per-fold sample_weight slices;
    the final refit sees the full weight vector.
    """
    recorded = []

    class WeightRecordingTree(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None):
            recorded.append((X.shape[1], None if sample_weight is None else len(sample_weight)))
            return super().fit(X, y, sample_weight=sample_weight)

    n_samples, n_features_total = X_train.shape
    weights = np.linspace(0.5, 1.5, n_samples)
    selector = GAFeatureSelectionCV(
        WeightRecordingTree(random_state=0),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=2,
        verbose=False,
    )

    selector.fit(X_train, y_train, sample_weight=weights)

    assert recorded, "no fit calls were recorded"
    # Every fit received weights, and X never grew beyond the original features.
    assert all(w is not None for _, w in recorded)
    assert all(f <= n_features_total for f, _ in recorded)
    # Candidate CV fits get per-fold weight slices; the refit gets all samples.
    assert all(w < n_samples for _, w in recorded[:-1])
    assert recorded[-1][1] == n_samples


@pytest.mark.parametrize("log_candidates", [False, True])
@pytest.mark.parametrize("final_selection", [False, True, "reuse_folds"])
def test_feature_search_uses_independent_fold_clones(log_candidates, final_selection):
    clones, fitted, logged = [], [], []

    class RecordingTree(DecisionTreeClassifier):
        def __sklearn_clone__(self):
            result = super().__sklearn_clone__()
            clones.append(result)
            return result

        def fit(self, X, y, sample_weight=None):
            fitted.append(self)
            return super().fit(X, y, sample_weight=sample_weight)

    class RecordLog:
        def create_run(self, parameters, score, estimator):
            logged.append(estimator)

    estimator = RecordingTree(random_state=7).fit(X_train, y_train)
    original_predictions = estimator.predict(X_test)
    fitted.clear()
    search_folds = list(StratifiedKFold(2).split(X_train, y_train))
    other_folds = list(StratifiedKFold(2, shuffle=True, random_state=3).split(X_train, y_train))
    search = GAFeatureSelectionCV(
        estimator,
        cv=search_folds,
        final_selection=final_selection,
        final_selection_cv=[search_folds[0], *other_folds],
        final_selection_top_k=2,
        scoring="accuracy",
        error_score="raise",
        population_size=4,
        generations=0,
        max_features=3,
        use_cache=False,
        n_jobs=1,
        random_state=7,
        verbose=False,
        log_config=RecordLog() if log_candidates else None,
    ).fit(X_train, y_train)

    counts = search.fit_counts_
    assert bool(counts["finalist_fold_fits"]) == bool(final_selection)
    assert bool(counts["finalist_reused_folds"]) == (final_selection == "reuse_folds")
    assert len(fitted) == counts["search_fold_fits"] + counts["finalist_fold_fits"] + 1
    assert counts["refit_fits"] == 1
    assert len(set(fitted)) == len(fitted)
    assert len(logged) == counts["search_cv_calls"] * log_candidates
    assert set(clones) == set(fitted) | set(logged)
    assert not set(fitted).intersection(logged)
    assert estimator not in fitted and estimator not in logged
    assert search.best_estimator_ is fitted[-1]
    np.testing.assert_array_equal(estimator.predict(X_test), original_predictions)


def test_feature_selection_unsupported_fit_param_raises_clearly():
    """#364: GAFeatureSelectionCV rejects unsupported metadata up front with a
    clear message naming the parameter and the wrapped estimator."""
    selector = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=0),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=1,
        error_score="raise",
        verbose=False,
    )

    with pytest.raises(TypeError) as excinfo:
        selector.fit(X_train, y_train, not_a_real_fit_param=np.ones(X_train.shape[0]))

    message = str(excinfo.value)
    assert "not_a_real_fit_param" in message
    assert "DecisionTreeClassifier" in message


@pytest.mark.parametrize(
    "method", ["predict", "predict_proba", "predict_log_proba", "decision_function", "score"]
)
@pytest.mark.parametrize("fitted_without_refit", [False, True])
def test_prediction_methods_require_refitted_winner(method, fitted_without_refit):
    from sklearn.linear_model import LogisticRegression

    selector = GAFeatureSelectionCV(
        LogisticRegression(),
        population_size=2,
        generations=0,
        cv=2,
        n_jobs=1,
        verbose=False,
        refit=not fitted_without_refit,
        random_state=0,
    )
    if fitted_without_refit:
        selector.fit(X_train[:40, :2], y_train[:40])
    args = (X_test[:, :2], y_test) if method == "score" else (X_test[:, :2],)
    with pytest.raises(NotFittedError, match="GAFeatureSelectionCV instance is not fitted"):
        getattr(selector, method)(*args)


@pytest.mark.parametrize("criteria", ["max", "min"])
@pytest.mark.parametrize(
    "proxy_scores,final_scores,top_k,expected_counts,max_features",
    [
        pytest.param([2, 1], [1, 2], 3, [1, 1, 2], None, id="rank-reversal"),
        pytest.param([1, 2], [1, 1], 3, [2, 1, 1], None, id="final-tie"),
        pytest.param([1] * 5, [1, 2, 3, 4, 5], 2, [1, 5], None, id="proxy-tie"),
        pytest.param([1] * 5, [1, 2, 3, 4, 5], 1, [1], None, id="one-finalist"),
        pytest.param([2, 1, 1, 1, 1], [1, 2, 3, 4, 5], 3, [1, 2, 5], None, id="cutoff-tie"),
        pytest.param([2, 1, 1, 1, 1], [1, 2, 3, 4, 5], 2, [1, 2], None, id="single-cutoff"),
        pytest.param([2, 2, 2, 2, 1], [1, 2, 3, 4, 5], 2, [1, 4], None, id="rank-priority"),
        pytest.param([1] * 5, [1] * 5, 2, [1, 5], None, id="both-tied"),
        pytest.param([1] * 5, [1, 2, 3, 4, 5], 2, [1, 3], 3, id="bounded-mask"),
    ],
)
def test_feature_final_selection_rechecks_masks_and_refits_full_winner(
    criteria, proxy_scores, final_scores, top_k, expected_counts, max_features
):
    from sklearn.model_selection import KFold

    fits = []

    class RecordedTree(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None):
            self.training_rows_ = len(X)
            np.testing.assert_array_equal(sample_weight, X[:, 0] % 100 + 1)
            fits.append((len(X), X.shape[1]))
            return super().fit(X, y, sample_weight=sample_weight)

    rows = np.arange(40)
    values = np.column_stack([rows + 100 * i for i in range(len(proxy_scores))])
    labels = rows % 2
    full_cv = list(KFold(2, shuffle=True, random_state=2).split(values))
    from sklearn_genetic import subsample_cv

    reduced_cv = subsample_cv(full_cv, 0.2, stratify=labels, random_state=2)
    sign = 1 if criteria == "max" else -1

    def learning_curve(model, X, y):
        count = model.n_features_in_
        scores = final_scores if model.training_rows_ >= 20 else proxy_scores
        return sign * scores[count - 1]

    search = GAFeatureSelectionCV(
        RecordedTree(random_state=0),
        cv=reduced_cv,
        scoring=learning_curve,
        population_size=4,
        generations=0,
        random_state=0,
        n_jobs=1,
        verbose=False,
        criteria=criteria,
        max_features=max_features,
        final_selection=True,
        final_selection_top_k=top_k,
        final_selection_cv=full_cv,
    ).fit(values, labels, sample_weight=rows + 1)
    expected_features = max(expected_counts, key=lambda count: (final_scores[count - 1], -count))
    assert search.support_.sum() == expected_features
    assert search.best_score_ == sign * final_scores[expected_features - 1]
    assert search.final_selection_results_["changed"] == (expected_features != expected_counts[0])
    candidates = search.final_selection_results_["candidates"]
    assert [sum(item["params"]["features"]) for item in candidates] == expected_counts
    assert len({tuple(item["params"]["features"]) for item in candidates}) == len(expected_counts)
    assert sum(rows == 20 for rows, _ in fits) == 2 * len(expected_counts)
    assert fits[-1] == (40, expected_features)
    assert search.predict(values).shape == labels.shape


@pytest.mark.parametrize("criteria", ["max", "min"])
def test_cached_ties_keep_the_same_feature_finalists_when_a_mask_is_repeated(criteria):
    masks = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
    X = np.tile([1, 2, 4, 8], (12, 1))
    y = np.tile([0, 1], 6)
    sign = 1 if criteria == "max" else -1

    def score(estimator, X, y):
        # Two inner folds tie. Three finalist folds distinguish the masks.
        return 1.0 if len(X) == 6 else sign * float(sum(X[0]))

    def fit(repeated):
        def optimize(problem, evaluate, report):
            report([handle for _, handle in evaluate(masks)])
            if repeated is not None:
                report([handle for _, handle in evaluate([repeated])])

        return GAFeatureSelectionCV(
            DummyClassifier(),
            cv=2,
            scoring=score,
            population_size=4,
            generations=1,
            optimizer_backend=optimize,
            max_features=2,
            final_selection=True,
            final_selection_cv=3,
            final_selection_top_k=2,
            criteria=criteria,
            random_state=9,
            verbose=False,
        ).fit(X, y)

    reference = fit(None)
    repeated = fit(masks[2])
    assert repeated.fit_stats_["cache_hits"] == 1
    assert (
        repeated.fit_stats_["cross_validate_calls"] == reference.fit_stats_["cross_validate_calls"]
    )
    assert (
        repeated.final_selection_results_["candidates"]
        == reference.final_selection_results_["candidates"]
    )
    np.testing.assert_equal(repeated.get_support(indices=True), [2, 3])
    assert repeated.best_score_ == reference.best_score_ == sign * 12.0


@pytest.mark.parametrize("criteria", ["max", "min"])
def test_feature_finalists_remain_distinct_with_uncached_score_ties(criteria):
    full_mask_scores = 0
    sign = 1 if criteria == "max" else -1

    def changing_score(model, X, y):
        nonlocal full_mask_scores
        if model.n_features_in_ == 2:
            full_mask_scores += 1
            return sign * (3 if full_mask_scores <= 2 else 2)
        return sign * 2

    search = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=0),
        cv=2,
        scoring=changing_score,
        criteria=criteria,
        population_size=8,
        generations=0,
        use_cache=False,
        random_state=0,
        n_jobs=1,
        verbose=False,
        final_selection=True,
        final_selection_top_k=3,
    ).fit(X_train[:40, :2], y_train[:40])
    full_proxy_scores = {
        score
        for mask, score in zip(
            search.cv_results_["features"], search.cv_results_["mean_test_score"]
        )
        if sum(mask) == 2
    }
    assert full_proxy_scores == {sign * 2, sign * 3}
    candidates = search.final_selection_results_["candidates"]
    assert len(candidates) == 3
    assert len({tuple(item["params"]["features"]) for item in candidates}) == 3
    assert search.support_.sum() == 1


@pytest.mark.parametrize("kind", ["hpo", "features"])
@pytest.mark.parametrize("all_failed", [False, True])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_final_selection_excludes_nonfinite_scores(kind, all_failed, invalid):
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer

    class RowCountTree(DecisionTreeClassifier):
        def fit(self, X, y, sample_weight=None):
            self.training_rows_ = len(X)
            return super().fit(X, y, sample_weight=sample_weight)

    values = np.arange(80).reshape(40, 2)
    labels = np.arange(40) % 2
    reduced_cv = [(np.arange(4), np.arange(20, 40))]
    full_cv = [(np.arange(20), np.arange(20, 40))]

    def learning_curve(model, X, y):
        size = model.max_depth if kind == "hpo" else model.n_features_in_
        if model.training_rows_ == 20:
            return invalid if all_failed or size == 1 else 1.0
        return 3 - size

    common = dict(
        cv=reduced_cv,
        scoring=learning_curve,
        population_size=4,
        generations=0,
        random_state=0,
        n_jobs=1,
        verbose=False,
        optimization_config=OptimizationConfig(
            final_selection=True, final_selection_top_k=3, final_selection_cv=full_cv
        ),
    )
    model = RowCountTree(max_depth=1, random_state=0)
    search = (
        GASearchCV(model, param_grid={"max_depth": Integer(1, 2)}, **common)
        if kind == "hpo"
        else GAFeatureSelectionCV(model, **common)
    )
    search = clone(search)
    if all_failed:
        with pytest.raises(ValueError, match="All final-selection candidates"):
            search.fit(values, labels)
    else:
        search.fit(values, labels)
        assert search.best_score_ == 1.0
        if kind == "hpo":
            assert search.best_params_["max_depth"] == 2
        else:
            assert search.support_.tolist() == [True, True]


@pytest.mark.parametrize("kind", ["hpo", "features"])
@pytest.mark.parametrize("top_k", [True, 1.5, np.nan])
def test_final_selection_requires_a_positive_integer_budget(kind, top_k):
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer

    cls = GASearchCV if kind == "hpo" else GAFeatureSelectionCV
    extra = {"param_grid": {"max_depth": Integer(1, 2)}} if kind == "hpo" else {}
    with pytest.raises(ValueError, match="final_selection_top_k must be"):
        cls(DecisionTreeClassifier(), final_selection_top_k=top_k, **extra)


@pytest.mark.parametrize("fraction", [0.25, 0.99, 1.0])
def test_subsample_cv_preserves_validation_and_reproducible_training(fraction):
    from sklearn_genetic import subsample_cv
    from sklearn.model_selection import StratifiedKFold

    labels = np.tile([0, 1, 2], 40)
    splits = list(StratifiedKFold(3).split(np.zeros((120, 1)), labels))
    sampled = subsample_cv(splits, fraction, stratify=labels, random_state=4)
    repeated = subsample_cv(splits, fraction, stratify=labels, random_state=4)
    assert len(sampled) == len(splits)
    for (train, test), (small, validation), (again, _) in zip(splits, sampled, repeated):
        np.testing.assert_array_equal(validation, test)
        np.testing.assert_array_equal(small, again)
        assert len(small) == int(len(train) * fraction)
        assert len(np.unique(small)) == len(small)
        assert set(small).issubset(train)
        assert not set(small).intersection(test)
        counts = np.bincount(labels[small], minlength=3)
        assert counts.min() >= len(small) // 3
        assert counts.max() <= (len(small) + 2) // 3
        if fraction == 1:
            np.testing.assert_array_equal(small, train)


@pytest.mark.parametrize("fraction", [0, 1.1, np.nan, True])
def test_subsample_cv_rejects_invalid_training_fraction(fraction):
    from sklearn_genetic import subsample_cv

    with pytest.raises(ValueError, match="train_size must be"):
        subsample_cv([(np.arange(10), np.arange(10, 15))], fraction)


@pytest.mark.parametrize("splits", [[(np.array([True, False, True]), [1])], []])
def test_subsample_cv_rejects_invalid_splits(splits):
    from sklearn_genetic import subsample_cv

    with pytest.raises(ValueError, match="cv_splits must contain"):
        subsample_cv(splits, 0.5)


def test_subsample_cv_rejects_loss_of_rare_class():
    from sklearn_genetic import subsample_cv

    labels = np.array([0] * 80 + [1] * 2)
    with pytest.raises(ValueError, match="retain every class"):
        subsample_cv([(np.arange(82), np.arange(82, 90))], 0.1, stratify=labels, random_state=4)


def test_subsample_cv_rejects_multidimensional_labels():
    from sklearn_genetic import subsample_cv

    with pytest.raises(ValueError, match="one-dimensional class labels"):
        subsample_cv(
            [(np.arange(10), np.arange(10, 15))], 0.5, stratify=np.zeros((15, 2)), random_state=4
        )


@pytest.mark.parametrize("disabled_option", ["final_selection", "refit"])
def test_feature_refit_clears_inactive_finalist_results(disabled_option):
    search = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=2),
        cv=2,
        scoring="accuracy",
        population_size=4,
        generations=0,
        random_state=2,
        n_jobs=1,
        verbose=False,
        final_selection=True,
        final_selection_top_k=2,
    ).fit(X_train, y_train)
    assert search.final_selection_results_["enabled"]
    search.set_params(**{disabled_option: False})
    search.fit(X_train, y_train)
    assert not hasattr(search, "final_selection_results_")
    if disabled_option == "refit":
        assert not hasattr(search, "best_estimator_")
        assert search.fit_counts_["refit_fits"] == 0
    index = search.best_index_
    np.testing.assert_equal(search.cv_results_["features"][index], search.support_)
    assert search.best_score_ == search.cv_results_["mean_test_score"][index]


@pytest.mark.parametrize(
    "algorithm,maximum,initializer,population_size",
    [
        (algorithm, maximum, "smart", 4)
        for algorithm in (
            "eaSimple",
            "eaMuPlusLambda",
            "eaMuCommaLambda",
            "eaDifferentialEvolution",
        )
        for maximum in (None, 2)
    ]
    + [("eaMuPlusLambda", maximum, "random", 4) for maximum in (None, 2)]
    + [("eaMuPlusLambda", None, "smart", 8)],
)
def test_timed_feature_search_evaluates_dense_initial_anchor(
    algorithm, maximum, initializer, population_size, monkeypatch
):
    from ..callbacks import early_stoppers

    ticks = [0]

    original_cv = genetic_search.cross_validate

    def timed_cv(*args, **kwargs):
        result = original_cv(*args, **kwargs)
        ticks[0] += 6
        return result

    monkeypatch.setattr(early_stoppers, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(genetic_search, "cross_validate", timed_cv)
    values, labels = load_iris(return_X_y=True)
    settings = dict(
        population_size=population_size,
        generations=0,
        population_initializer=initializer,
        algorithm=algorithm,
        max_features=maximum,
        cv=2,
        n_jobs=1,
        random_state=31,
        verbose=False,
        error_score="raise",
    )
    reference = GAFeatureSelectionCV(DecisionTreeClassifier(random_state=0), **settings)
    reference.fit(values, labels)
    masks = reference.cv_results_["features"]
    expected = max(masks, key=np.sum) if initializer == "smart" else masks[0]
    if initializer == "smart":
        assert np.sum(masks[0]) == 1
        assert np.sum(expected) == (maximum or values.shape[1])

    timed = GAFeatureSelectionCV(DecisionTreeClassifier(random_state=0), **settings)
    timed.fit(values, labels, callbacks=TimerStopping(5))
    assert timed.fit_stats_["cross_validate_calls"] == 1
    assert len(timed.cv_results_["features"]) == 1
    np.testing.assert_array_equal(timed.support_, expected)
    assert timed.support_.any()
    assert timed.predict(values).shape == labels.shape

    # With enough time, the same candidate masks and scores are still evaluated.
    timed.fit(values, labels, callbacks=TimerStopping(1000))
    results = lambda search: sorted(
        (tuple(mask), score)
        for mask, score in zip(
            search.cv_results_["features"], search.cv_results_["mean_test_score"]
        )
    )
    assert results(timed) == results(reference)
    np.testing.assert_array_equal(timed.cv_results_["features"][0], expected)


def test_timed_feature_resume_preserves_initial_candidate_order(tmp_path):
    import shutil
    from pathlib import Path
    from ..callbacks import ModelCheckpoint

    values, labels = load_iris(return_X_y=True)
    path = str(tmp_path / "feature_order.pkl")
    settings = dict(
        population_size=4,
        generations=0,
        cv=2,
        n_jobs=1,
        random_state=31,
        use_cache=False,
        verbose=False,
        error_score="raise",
    )
    original = GAFeatureSelectionCV(DecisionTreeClassifier(random_state=0), **settings)
    original.fit(values, labels, callbacks=ModelCheckpoint(path))
    assert Path(path).is_file()
    masks = original.cv_results_["features"]
    assert np.sum(masks[0]) == 1

    reference_path = tmp_path / "untimed.pkl"
    shutil.copyfile(path, reference_path)
    reference = GAFeatureSelectionCV(DecisionTreeClassifier(random_state=0), **settings)
    reference.fit(values, labels, callbacks=ModelCheckpoint(reference_path))
    resumed = GAFeatureSelectionCV(DecisionTreeClassifier(random_state=0), **settings)
    resumed.fit(values, labels, callbacks=[ModelCheckpoint(path), TimerStopping(60)])
    assert len(resumed.cv_results_["features"]) == 2 * len(masks)
    np.testing.assert_array_equal(
        resumed.cv_results_["features"], reference.cv_results_["features"]
    )
    np.testing.assert_array_equal(
        resumed.cv_results_["mean_test_score"],
        reference.cv_results_["mean_test_score"],
    )


@pytest.mark.parametrize("features", [False, True])
@pytest.mark.parametrize("complete_overlap", [False, True])
def test_final_selection_reuse_trains_only_missing_folds(features, complete_overlap):
    from sklearn.model_selection import KFold
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer

    fit_sizes = []

    class RecordedTree(DecisionTreeClassifier):
        def fit(self, X, y):
            fit_sizes.append(len(y))
            self.first_row_ = X[0, 0] % 100
            return super().fit(X, y)

    def score(model, X, y):
        return (
            model.first_row_
            + X[0, 0] % 100 / 100
            + (model.max_depth or sum(X[0] // 100 + 1) ** 2 + X.shape[1])
        )

    rows = np.arange(36)
    X = np.column_stack([rows + 100 * i for i in range(4)])
    y = rows % 2
    folds = list(KFold(3).split(X))
    final = [
        (train.astype("int32"), test.astype("int32"))
        for train, test in [folds[2], folds[0], folds[1]]
    ]
    preliminary = (
        folds
        if complete_overlap
        else [folds[0], (folds[1][0][::-1], folds[1][1]), (folds[2][0], folds[2][1][::-1])]
    )
    runs = []
    for mode in [True, "reuse_folds"]:
        options = dict(
            cv=preliminary,
            final_selection_cv=final,
            scoring=score,
            n_jobs=1,
            error_score="raise",
            random_state=7,
            population_size=5,
            generations=0,
            verbose=False,
            optimization_config=OptimizationConfig(final_selection=mode),
        )
        model = RecordedTree(random_state=0)
        search = (
            GAFeatureSelectionCV(model, **options)
            if features
            else GASearchCV(model, param_grid={"max_depth": Integer(1, 4)}, **options)
        )
        fit_sizes.clear()
        search.fit(X, y)
        runs.append((search, len(fit_sizes)))

    (baseline, before), (optimized, after) = runs
    candidates = optimized.final_selection_results_["candidates"]
    reused = 3 if complete_overlap else 1
    assert len({tuple(c["cv_scores"]) for c in candidates}) > 1
    assert before - after == len(candidates) * reused > 0
    assert [c["reused_folds"] for c in candidates] == [reused] * len(candidates)
    assert [c["cv_scores"] for c in candidates] == [
        c["cv_scores"] for c in baseline.final_selection_results_["candidates"]
    ]
    for fitted, actual_fits, reused_per_finalist in (
        (baseline, before, 0),
        (optimized, after, reused),
    ):
        calls = fitted.fit_stats_["cross_validate_calls"]
        expected = {
            "search_cv_calls": calls,
            "search_fold_fits": calls * 3,
            "finalist_cv_calls": len(candidates) if reused_per_finalist < 3 else 0,
            "finalist_fold_fits": len(candidates) * (3 - reused_per_finalist),
            "finalist_reused_folds": len(candidates) * reused_per_finalist,
            "refit_fits": 1,
        }
        assert fitted.fit_counts_ == expected
        assert (
            sum(expected[key] for key in ("search_fold_fits", "finalist_fold_fits", "refit_fits"))
            == actual_fits
        )
    assert optimized.best_index_ == baseline.best_index_
    np.testing.assert_array_equal(optimized.predict(X), baseline.predict(X))


def test_finalist_fit_counts_include_a_partial_attempt_followed_by_full_validation():
    train, test = np.arange(12), np.arange(12, 24)
    preliminary = [(test, train)]
    fits = []

    class RecordedTree(DecisionTreeClassifier):
        def fit(self, X, y):
            fits.append(len(y))
            return super().fit(X, y)

    def mixed_score(model, X, y):
        return 2**53 + 1 if X[0, 0] else 0.5

    rows = np.arange(24)
    search = GAFeatureSelectionCV(
        RecordedTree(random_state=0),
        cv=preliminary,
        final_selection_cv=[*preliminary, (train, test)],
        final_selection="reuse_folds",
        final_selection_top_k=1,
        error_score="raise",
        scoring=mixed_score,
        population_size=3,
        generations=0,
        random_state=2,
        n_jobs=1,
        verbose=False,
    ).fit(np.column_stack([rows, rows]), rows % 2)

    counts = search.fit_counts_
    assert counts["finalist_cv_calls"] == 2
    assert counts["finalist_fold_fits"] == 3
    assert counts["finalist_reused_folds"] == 0
    assert len(fits) == counts["search_fold_fits"] + 3 + counts["refit_fits"]


def test_finalist_reuse_checks_all_indices_when_test_fold_keys_collide():
    rows = np.arange(24)
    X, y = np.column_stack([rows, rows]), rows % 2

    def split(test):
        return np.setdiff1d(rows, test), np.array(test)

    preliminary = [split([0, 1, 5]), split([0, 3, 5])]
    train, test = preliminary[1]
    final = [(train[::-1], test), (train, test), split([0, 2, 5])]
    optimized = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=2),
        cv=preliminary,
        final_selection_cv=final,
        final_selection="reuse_folds",
        final_selection_top_k=1,
        scoring="accuracy",
        error_score="raise",
        population_size=3,
        generations=0,
        random_state=2,
        n_jobs=1,
        verbose=False,
    ).fit(X, y)
    baseline = clone(optimized).set_params(final_selection=True).fit(X, y)

    assert optimized.fit_counts_["finalist_reused_folds"] == 1
    assert optimized.fit_counts_["finalist_fold_fits"] == 2
    assert optimized.best_score_ == baseline.best_score_
    np.testing.assert_equal(optimized.predict(X), baseline.predict(X))


@pytest.mark.parametrize("case", ["mutated", "malformed", "large_integer", "error_score"])
def test_final_selection_reuse_preserves_split_and_score_contracts(case):
    train, test = np.arange(12), np.arange(12, 24)
    search_folds = [(train, test), (test, train)]

    class FinalSplit:
        def split(self, X, y):
            if case == "mutated":
                test[:] = test[::-1]
            yield (train, test, test) if case == "malformed" else (train, test)

    def first_row_score(model, X, y):
        if case == "large_integer":
            return 2**53 + 1 if X[0, 0] else 0.5
        return int(X[0, 0])

    search = GAFeatureSelectionCV(
        DecisionTreeClassifier(random_state=0),
        cv=search_folds,
        final_selection_cv=FinalSplit(),
        final_selection="reuse_folds",
        error_score=np.nan if case == "error_score" else "raise",
        scoring=first_row_score,
        population_size=3,
        generations=0,
        random_state=2,
        n_jobs=1,
        verbose=False,
    )
    rows = np.arange(24)
    X = np.column_stack([rows, rows])
    if case in ("malformed", "error_score"):
        with pytest.raises(
            ValueError, match="too many values to unpack" if case == "malformed" else "error_score"
        ):
            search.fit(X, rows % 2)
    else:
        search.fit(X, rows % 2)
        candidates = search.final_selection_results_["candidates"]
        expected = 2**53 + 1 if case == "large_integer" else 23
        assert candidates and all(
            c["cv_scores"] == [expected] and c["reused_folds"] == 0 for c in candidates
        )
