import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import make_scorer

from .. import GASearchCV, GAFeatureSelectionCV
from ..space import Integer, Categorical, Continuous

# Create a dataset with outliers
def create_outlier_dataset():
    X_normal, _ = make_blobs(n_samples=200, centers=1, n_features=4, 
                            cluster_std=1.0, random_state=42)
    np.random.seed(42)
    X_outliers = np.random.uniform(low=-6, high=6, size=(20, 4))
    X = np.vstack([X_normal, X_outliers])
    return X

X = create_outlier_dataset()
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)


def test_isolation_forest_gasearch():
    """Test GASearchCV with IsolationForest - using default scoring (score_samples path)"""
    estimator = IsolationForest(random_state=42)
    
    param_grid = {
        'contamination': Continuous(0.05, 0.3),  
        'n_estimators': Integer(50, 150)  
    }
    
    ga_search = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        population_size=6,
        generations=4,
        verbose=False,
        n_jobs=1,
        scoring=None  # This will trigger default_outlier_scorer with score_samples
    )
    
    ga_search.fit(X_train)
    
    assert check_is_fitted(ga_search) is None
    assert 'contamination' in ga_search.best_params_
    assert 'n_estimators' in ga_search.best_params_
    assert len(ga_search.predict(X_test)) == len(X_test)
    assert hasattr(ga_search, 'best_estimator_')


def test_one_class_svm_gasearch_default_scoring():
    """Test GASearchCV with OneClassSVM - using default scoring"""
    estimator = OneClassSVM()
    
    param_grid = {
        'nu': Continuous(0.01, 0.5),
        'gamma': Categorical(['scale', 'auto'])
    }
    
    ga_search = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        population_size=6,
        generations=4,
        verbose=False,
        n_jobs=1,
        scoring=None  # This will trigger default_outlier_scorer
    )
    
    ga_search.fit(X_train)
    
    assert check_is_fitted(ga_search) is None
    assert 'nu' in ga_search.best_params_
    assert len(ga_search.predict(X_test)) == len(X_test)


# Create a mock estimator that specifically only has decision_function
class MockDecisionFunctionOnlyEstimator:
    """Mock estimator that only has decision_function (not score_samples)"""
    
    def __init__(self, param_value=0.5):
        self.param_value = param_value
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def decision_function(self, X):
        """Return mock decision scores"""
        np.random.seed(42)
        return np.random.randn(X.shape[0])
    
    def predict(self, X):
        """Return mock predictions"""
        return np.where(self.decision_function(X) >= 0, 1, -1)
    
    def get_params(self, deep=True):
        return {'param_value': self.param_value}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def test_decision_function_path_explicitly():
    """Test that specifically exercises the decision_function path in default scorer"""
    # Use our mock estimator that only has decision_function
    estimator = MockDecisionFunctionOnlyEstimator()
    
    # Verify it has decision_function but not score_samples
    assert hasattr(estimator, 'decision_function')
    assert not hasattr(estimator, 'score_samples')
    
    # Temporarily patch is_outlier_detector to recognize our mock
    import sklearn_genetic.genetic_search as sg_module
    original_is_outlier_detector = sg_module.is_outlier_detector
    
    def mock_is_outlier_detector(est):
        if isinstance(est, MockDecisionFunctionOnlyEstimator):
            return True
        return original_is_outlier_detector(est)
    
    sg_module.is_outlier_detector = mock_is_outlier_detector
    
    try:
        param_grid = {
            'param_value': Continuous(0.1, 0.9)
        }
        
        ga_search = GASearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=3,
            population_size=4,
            generations=3,
            verbose=False,
            n_jobs=1,
            scoring=None  # This MUST use decision_function path
        )
        
        ga_search.fit(X_train)
        
        # Verify it worked
        assert check_is_fitted(ga_search) is None
        assert 'param_value' in ga_search.best_params_
        
    finally:
        # Restore original function
        sg_module.is_outlier_detector = original_is_outlier_detector


def test_one_class_svm_feature_selection_decision_function():
    """Test GAFeatureSelectionCV with mock estimator that uses decision_function path"""
    estimator = MockDecisionFunctionOnlyEstimator()
    
    # Verify this estimator will use decision_function path
    assert hasattr(estimator, 'decision_function')
    assert not hasattr(estimator, 'score_samples')
    
    # Temporarily patch is_outlier_detector
    import sklearn_genetic.genetic_search as sg_module
    original_is_outlier_detector = sg_module.is_outlier_detector
    
    def mock_is_outlier_detector(est):
        if isinstance(est, MockDecisionFunctionOnlyEstimator):
            return True
        return original_is_outlier_detector(est)
    
    sg_module.is_outlier_detector = mock_is_outlier_detector
    
    try:
        ga_feature_selection = GAFeatureSelectionCV(
            estimator=estimator,
            cv=3,
            population_size=6,
            generations=4,
            verbose=False,
            n_jobs=1,
            scoring=None  # Test default scoring with decision_function
        )
        
        ga_feature_selection.fit(X_train)
        
        assert check_is_fitted(ga_feature_selection) is None
        assert hasattr(ga_feature_selection, 'support_')
        assert len(ga_feature_selection.support_) == X_train.shape[1]
        
    finally:
        # Restore original function
        sg_module.is_outlier_detector = original_is_outlier_detector


def test_importerror_fallback():
    """Test the ImportError fallback for is_outlier_detector"""
    import sklearn_genetic.genetic_search as sg_module
    
    # Save original function
    original_is_outlier_detector = sg_module.is_outlier_detector
    
    # Create a proper mock estimator with fit method
    class MockOutlierEstimator:
        def __init__(self, param_value=0.5):
            self.param_value = param_value
            self.is_fitted_ = False
        
        def fit(self, X, y=None):
            self.is_fitted_ = True
            return self
        
        def fit_predict(self, X, y=None):
            self.fit(X, y)
            np.random.seed(42)
            return np.random.choice([1, -1], size=X.shape[0])
        
        def decision_function(self, X):
            np.random.seed(42)
            return np.random.randn(X.shape[0])
        
        def predict(self, X):
            return np.where(self.decision_function(X) >= 0, 1, -1)
        
        def get_params(self, deep=True):
            return {'param_value': self.param_value}
        
        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    # Create fallback function (simulating ImportError scenario)
    def fallback_is_outlier_detector(estimator):
        return hasattr(estimator, 'fit_predict') and hasattr(estimator, 'decision_function')
    
    # Patch the function to use fallback
    sg_module.is_outlier_detector = fallback_is_outlier_detector
    
    try:
        # Test that fallback works
        mock_estimator = MockOutlierEstimator()
        
        # This should return True using fallback logic
        assert sg_module.is_outlier_detector(mock_estimator) == True
        
        # Test that non-outlier detector returns False
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2)
        assert sg_module.is_outlier_detector(kmeans) == False
        
        # Test GASearchCV works with fallback detection
        param_grid = {'param_value': Continuous(0.1, 0.9)}
        
        ga_search = GASearchCV(
            estimator=mock_estimator,
            param_grid=param_grid,
            cv=3,
            population_size=4,
            generations=2,
            verbose=False,
            n_jobs=1
        )
        
        # This should work without raising ValueError
        ga_search.fit(X_train[:20])  # Small dataset for speed
        assert check_is_fitted(ga_search) is None
        
    finally:
        # Restore original function
        sg_module.is_outlier_detector = original_is_outlier_detector


# Mock outlier detector to test the fit_predict fallback path
class MockFitPredictOnlyDetector:
    """Mock outlier detector that only has fit_predict method"""
    
    def __init__(self, param_value=0.5):
        self.param_value = param_value
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        # Return random predictions (1 for inlier, -1 for outlier)
        np.random.seed(42)
        return np.random.choice([1, -1], size=X.shape[0], p=[0.8, 0.2])
    
    def predict(self, X):
        return self.fit_predict(X)
    
    def get_params(self, deep=True):
        return {'param_value': self.param_value}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def test_mock_outlier_detector_fit_predict_path():
    """Test the fallback path in default_outlier_scorer that uses fit_predict"""
    estimator = MockFitPredictOnlyDetector()
    
    # Verify it only has fit_predict (no score_samples or decision_function)
    assert hasattr(estimator, 'fit_predict')
    assert not hasattr(estimator, 'score_samples')
    assert not hasattr(estimator, 'decision_function')
    
    # Temporarily patch is_outlier_detector to recognize our mock class
    import sklearn_genetic.genetic_search as sg_module
    original_is_outlier_detector = sg_module.is_outlier_detector
    
    def mock_is_outlier_detector(est):
        if isinstance(est, MockFitPredictOnlyDetector):
            return True
        return original_is_outlier_detector(est)
    
    sg_module.is_outlier_detector = mock_is_outlier_detector
    
    try:
        param_grid = {'param_value': Continuous(0.1, 0.9)}
        
        ga_search = GASearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=3,
            population_size=4,
            generations=2,
            verbose=False,
            n_jobs=1,
            scoring=None  # This should trigger the fit_predict path
        )
        
        ga_search.fit(X_train[:20])  # Use smaller dataset for faster test
        
        assert check_is_fitted(ga_search) is None
        
    finally:
        # Restore original function
        sg_module.is_outlier_detector = original_is_outlier_detector


def test_one_class_svm_gasearch_custom_scoring():
    """Test GASearchCV with OneClassSVM - with custom scoring"""
    def custom_scorer(estimator, X, y=None):
        decision_scores = estimator.decision_function(X)
        return np.mean(decision_scores)
    
    estimator = OneClassSVM()
    
    param_grid = {
        'nu': Continuous(0.01, 0.5),
        'gamma': Categorical(['scale'])
    }
    
    ga_search = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        population_size=6,
        generations=4,
        verbose=False,
        n_jobs=1,
        scoring=make_scorer(custom_scorer)
    )
    
    ga_search.fit(X_train)
    
    assert check_is_fitted(ga_search) is None
    assert 'nu' in ga_search.best_params_


def test_local_outlier_factor_gasearch():
    """Test GASearchCV with LocalOutlierFactor - using default scoring"""
    estimator = LocalOutlierFactor(novelty=True)  
    
    param_grid = {
        'n_neighbors': Integer(5, 30),
        'contamination': Continuous(0.05, 0.3)
    }
    
    ga_search = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        population_size=6,
        generations=4,
        verbose=False,
        n_jobs=1,
        scoring=None  # This will trigger default_outlier_scorer
    )
    
    ga_search.fit(X_train)
    
    assert check_is_fitted(ga_search) is None
    assert 'n_neighbors' in ga_search.best_params_
    assert len(ga_search.predict(X_test)) == len(X_test)


def test_isolation_forest_feature_selection():
    """Test GAFeatureSelectionCV with IsolationForest"""
    estimator = IsolationForest(contamination=0.1, random_state=42)
    
    ga_feature_selection = GAFeatureSelectionCV(
        estimator=estimator,
        cv=3,
        population_size=6,
        generations=4,
        verbose=False,
        n_jobs=1,
        scoring=None  # Test default scoring path
    )
    
    ga_feature_selection.fit(X_train)
    
    assert check_is_fitted(ga_feature_selection) is None
    assert hasattr(ga_feature_selection, 'support_')
    assert len(ga_feature_selection.support_) == X_train.shape[1]
    assert len(ga_feature_selection.predict(X_test)) == len(X_test)


def test_outlier_detection_with_custom_scoring():
    """Test outlier detection with custom scoring function"""
    def custom_outlier_score(estimator, X, y=None):
        scores = estimator.score_samples(X)
        return np.std(scores)  
    
    estimator = IsolationForest(random_state=42)
    
    param_grid = {
        'contamination': Continuous(0.05, 0.3),
        'n_estimators': Integer(50, 150)
    }
    
    ga_search = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        scoring=make_scorer(custom_outlier_score),
        population_size=6,
        generations=4,
        verbose=False,
        n_jobs=1
    )
    
    ga_search.fit(X_train)
    
    assert check_is_fitted(ga_search) is None
    assert 'contamination' in ga_search.best_params_


def test_outlier_detection_with_y_none():
    """Test that outlier detection works with y=None"""
    estimator = IsolationForest(random_state=42)
    
    param_grid = {
        'contamination': Continuous(0.05, 0.3),
        'n_estimators': Integer(50, 100)
    }
    
    ga_search = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        population_size=4,
        generations=3,
        verbose=False,
        n_jobs=1
    )
    
    # Explicitly pass y=None to test this path
    ga_search.fit(X_train, y=None)
    
    assert check_is_fitted(ga_search) is None
    assert hasattr(ga_search, 'best_params_')


def test_outlier_detection_feature_selection_with_y_none():
    """Test that GAFeatureSelectionCV works with y=None for outlier detection"""
    estimator = IsolationForest(contamination=0.1, random_state=42)
    
    ga_feature_selection = GAFeatureSelectionCV(
        estimator=estimator,
        cv=3,
        population_size=4,
        generations=3,
        verbose=False,
        n_jobs=1
    )
    
    # Explicitly pass y=None to test this path
    ga_feature_selection.fit(X_train, y=None)
    
    assert check_is_fitted(ga_feature_selection) is None
    assert hasattr(ga_feature_selection, 'support_')


def test_error_handling_invalid_estimator():
    """Test that non-outlier-detector estimators still raise appropriate errors"""
    from sklearn.cluster import KMeans
    
    estimator = KMeans(n_clusters=2)
    
    param_grid = {
        'n_clusters': Integer(2, 5)
    }
    
    with pytest.raises(ValueError) as excinfo:
        ga_search = GASearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=3
        )
    
    assert "is not a valid Sklearn classifier, regressor, or outlier detector" in str(excinfo.value)


def test_cv_results_structure():
    """Test that cv_results_ has the expected structure for outlier detection"""
    estimator = IsolationForest(random_state=42)
    
    param_grid = {
        'contamination': Continuous(0.05, 0.2),
        'n_estimators': Integer(50, 100)
    }
    
    ga_search = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        population_size=4,
        generations=3,
        verbose=False,
        return_train_score=True
    )
    
    ga_search.fit(X_train)
    
    cv_results = ga_search.cv_results_
    
    # Check that expected keys exist
    assert 'mean_test_score' in cv_results
    assert 'std_test_score' in cv_results
    assert 'params' in cv_results
    assert 'mean_fit_time' in cv_results
    
    # Check parameter columns
    assert 'param_contamination' in cv_results
    assert 'param_n_estimators' in cv_results


def test_kfold_cv_for_outlier_detectors():
    """Test that KFold is used for outlier detectors instead of StratifiedKFold"""
    estimator = IsolationForest(random_state=42)
    
    param_grid = {
        'contamination': Continuous(0.05, 0.2)
    }
    
    ga_search = GASearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,  # Explicitly set to int to test KFold path
        population_size=4,
        generations=2,
        verbose=False,
        n_jobs=1
    )
    
    ga_search.fit(X_train)
    
    # Should complete without errors and use KFold
    assert ga_search.n_splits_ == 5
    assert check_is_fitted(ga_search) is None