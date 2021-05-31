from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from .. import GASearchCV
from ..plots import plot_fitness_evolution, plot_search_space
from ..space import Integer, Categorical, Continuous


data = load_boston()

y = data["target"]
X = data["data"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

clf = DecisionTreeRegressor()

evolved_estimator = GASearchCV(
    clf,
    cv=2,
    scoring="r2",
    population_size=8,
    generations=5,
    tournament_size=3,
    elitism=True,
    crossover_probability=0.9,
    mutation_probability=0.05,
    param_grid={
        "ccp_alpha": Continuous(0, 1),
        "criterion": Categorical(["mse", "mae"]),
        "max_depth": Integer(2, 20),
        "min_samples_split": Integer(2, 30),
    },
    criteria="max",
    n_jobs=-1,
)

evolved_estimator.fit(X_train, y_train)


def test_plot_evolution():
    plot = plot_fitness_evolution(evolved_estimator)


def test_plot_space():
    plot = plot_search_space(evolved_estimator)
    plot = plot_search_space(
        evolved_estimator, features=["ccp_alpha", "max_depth", "min_samples_split"]
    )
