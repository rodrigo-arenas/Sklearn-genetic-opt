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
    # Generate normal data
    X_normal, _ = make_blobs(n_samples=200, centers=1, n_features=4, 
                            cluster_std=1.0, random_state=42)
    
    # Add some outliers
    np.random.seed(42)
    X_outliers = np.random.uniform(low=-6, high=6, size=(20, 4))
    
    X = np.vstack([X_normal, X_outliers])
    return X

X = create_outlier_dataset()
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)


def test_isolation_forest_gasearch():
    """Test GASearchCV with IsolationForest - using default scoring"""
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
        scoring=None  
    )
    
    # Fit without y (unsupervised)
    ga_search.fit(X_train)
    
    assert check_is_fitted(ga_search) is None
    assert 'contamination' in ga_search.best_params_
    assert 'n_estimators' in ga_search.best_params_
    assert len(ga_search.predict(X_test)) == len(X_test)
    assert hasattr(ga_search, 'best_estimator_')


def test_one_class_svm_gasearch():
    """Test GASearchCV with OneClassSVM - with custom scoring"""
    def custom_scorer(estimator, X, y=None):
        # Custom scoring for OneClassSVM
        decision_scores = estimator.decision_function(X)
        return np.mean(decision_scores)
    
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
        scoring=make_scorer(custom_scorer)
    )
    
    ga_search.fit(X_train)
    
    assert check_is_fitted(ga_search) is None
    assert 'nu' in ga_search.best_params_
    assert len(ga_search.predict(X_test)) == len(X_test)


def test_local_outlier_factor_gasearch():
    """Test GASearchCV with LocalOutlierFactor"""
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
        n_jobs=1
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
        n_jobs=1
    )
    
    ga_feature_selection.fit(X_train)
    
    assert check_is_fitted(ga_feature_selection) is None
    assert hasattr(ga_feature_selection, 'support_')
    assert len(ga_feature_selection.support_) == X_train.shape[1]
    assert len(ga_feature_selection.predict(X_test)) == len(X_test)


def test_outlier_detection_with_custom_scoring():
    """Test outlier detection with custom scoring function"""
    def custom_outlier_score(estimator, X, y=None):
        """Custom scoring that measures the spread of anomaly scores"""
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