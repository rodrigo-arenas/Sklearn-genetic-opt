Using Adapters
==============

Introduction
------------

Adapters let you change the mutation and crossover probabilities as the
optimization progresses, instead of keeping them fixed for every generation.
In practice, an adapter is a small schedule that returns a new probability at
each generation.

This makes it possible to express different search strategies, for example:

* Start with a high mutation probability to explore more diverse solutions, then
  slowly reduce it to refine the most promising candidates.
* Start with a low crossover probability and increase it over time.
* Use different schedules for mutation and crossover.

All adapters use three parameters:

* **initial_value:** value used at generation 0.
* **end_value:** target value approached by the schedule.
* **adaptive_rate:** controls how quickly the schedule approaches ``end_value``.
  Larger values make the schedule change faster.

The following notation is used throughout this tutorial:

===================== ===============
Name                  Notation
===================== ===============
initial_value         :math:`p_0`
end_value             :math:`p_f`
current generation    :math:`t`
adaptive_rate         :math:`\alpha`
value at generation t :math:`p(t; \alpha)`
===================== ===============

The initial value :math:`p_0` does not need to be greater than the final value
:math:`p_f`.

If :math:`p_0 > p_f`, the adapter defines a decay toward :math:`p_f`.

If :math:`p_0 < p_f`, the adapter defines an ascent toward :math:`p_f`.

All non-constant adapters :math:`p(t; \alpha)`, for
:math:`\alpha \in (0,1)`, have the following properties:

.. math::

   \lim_{t \to 0^{+}} p(t; \alpha) = p_0\\
   \\
   \lim_{t \to +\infty} p(t; \alpha) = p_f

The following adapters are available:

* ConstantAdapter
* ExponentialAdapter
* InverseAdapter
* PotentialAdapter


ConstantAdapter
---------------

This adapter is mainly used internally by the package. When you pass the
crossover or mutation probability as a real number, the package converts that
number into a ``ConstantAdapter``. This lets the optimization code use the same
internal API for both fixed probabilities and scheduled probabilities.

Its definition is:

.. math::

   p(t; \alpha) = p_0


ExponentialAdapter
------------------

The ``ExponentialAdapter`` changes the probability exponentially:

.. math::

   p(t; \alpha) = (p_0-p_f)e^{-\alpha t} + p_f

Usage example:

.. code:: python3

    from sklearn_genetic.schedules import ExponentialAdapter

    # Decay from initial_value toward end_value
    adapter = ExponentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)

    # Run a few iterations
    for _ in range(3):
        adapter.step()  # 0.8, 0.74, 0.69

The following plots show the adapter for different values of
:math:`\alpha`. Larger values of :math:`\alpha` move toward the target value
more quickly.

**decay:**

.. image:: ../images/schedules_exponential_0.png

**ascent:**

.. image:: ../images/schedules_exponential_1.png

.. code:: python3

   import matplotlib.pyplot as plt
   from sklearn_genetic.schedules import ExponentialAdapter

   values = [{"initial_value": 0.8, "end_value": 0.2},  # Decay
             {"initial_value": 0.2, "end_value": 0.8}]  # Ascent
   alphas = [0.8, 0.4, 0.1, 0.05]

   for value in values:
       for alpha in alphas:
           adapter = ExponentialAdapter(**value, adaptive_rate=alpha)
           adapter_result = [adapter.step() for _ in range(100)]

           plt.plot(adapter_result, label=r'$\alpha={}$'.format(alpha))

       plt.xlabel(r'$t$')
       plt.ylabel(r'$p(t; \alpha)$')
       plt.title("Exponential Adapter")
       plt.legend()
       plt.show()


InverseAdapter
--------------

The ``InverseAdapter`` changes the probability with inverse decay:

.. math::

   p(t; \alpha) = \frac{(p_0-p_f)}{1+\alpha t} + p_f

Usage example:

.. code:: python3

    from sklearn_genetic.schedules import InverseAdapter

    # Decay from initial_value toward end_value
    adapter = InverseAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)

    # Run a few iterations
    for _ in range(3):
        adapter.step()  # 0.8, 0.75, 0.7

The following plots show the adapter for different values of
:math:`\alpha`.

**decay:**

.. image:: ../images/schedules_inverse_0.png

**ascent:**

.. image:: ../images/schedules_inverse_1.png

PotentialAdapter
----------------

The ``PotentialAdapter`` changes the probability with the following form:

.. math::

   p(t; \alpha) = (p_0-p_f)(1-\alpha)^{ t} + p_f

Usage example:

.. code:: python3

    from sklearn_genetic.schedules import PotentialAdapter

    # Decay from initial_value toward end_value
    adapter = PotentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)

    # Run a few iterations
    for _ in range(3):
        adapter.step()  # 0.8, 0.26, 0.206

The following plots show the adapter for different values of
:math:`\alpha`.

**decay:**

.. image:: ../images/schedules_potential_0.png

**ascent:**

.. image:: ../images/schedules_potential_1.png

Compare
-------

The following plots compare all adapters using the same value of
:math:`\alpha`.

**decay:**

.. image:: ../images/schedules_comparison_0.png

**ascent:**

.. image:: ../images/schedules_comparison_1.png

.. code:: python3

   import matplotlib.pyplot as plt
   from sklearn_genetic.schedules import ExponentialAdapter, PotentialAdapter, InverseAdapter


   params = {"initial_value": 0.2, "end_value": 0.8, "adaptive_rate": 0.15}  # Ascent
   adapters = [ExponentialAdapter(**params), PotentialAdapter(**params), InverseAdapter(**params)]

   for adapter in adapters:
       adapter_result = [adapter.step() for _ in range(50)]

       plt.plot(adapter_result, label=f"{type(adapter).__name__}")

   plt.xlabel(r'$t$')
   plt.ylabel(r'$p(t; \alpha)$')
   plt.title("Adapters Comparison")
   plt.legend()
   plt.show()


Full Example
------------

In this example, we create a decay schedule for the mutation probability and an
ascent schedule for the crossover probability. Let us call them
:math:`p_{mt}(t; \alpha)` and :math:`p_{cr}(t; \alpha)`, respectively.

This strategy encourages the optimizer to explore more diverse solutions in the
first generations, then rely more on crossover as the population improves.

When choosing :math:`\alpha`, :math:`p_0`, and :math:`p_f`, make sure the
mutation and crossover probabilities remain compatible with the evolutionary
algorithm. The implementation requires:

.. math::

   p_{mt}(t; \alpha) + p_{cr}(t; \alpha) <= 1;  \forall t

The same idea can be used for hyperparameter tuning or feature selection.

.. code-block:: python

   from sklearn_genetic import GASearchCV
   from sklearn_genetic.schedules import ExponentialAdapter
   from sklearn_genetic.space import Continuous, Categorical, Integer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split, StratifiedKFold
   from sklearn.datasets import load_digits
   from sklearn.metrics import accuracy_score

   data = load_digits()
   n_samples = len(data.images)
   X = data.images.reshape((n_samples, -1))
   y = data['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

   clf = RandomForestClassifier()

   mutation_adapter = ExponentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)
   crossover_adapter = ExponentialAdapter(initial_value=0.2, end_value=0.8, adaptive_rate=0.1)

   param_grid = {'min_weight_fraction_leaf': Continuous(0.01, 0.5, distribution='log-uniform'),
                 'bootstrap': Categorical([True, False]),
                 'max_depth': Integer(2, 30),
                 'max_leaf_nodes': Integer(2, 35),
                 'n_estimators': Integer(100, 300)}

   cv = StratifiedKFold(n_splits=3, shuffle=True)

   evolved_estimator = GASearchCV(estimator=clf,
                                  cv=cv,
                                  scoring='accuracy',
                                  population_size=20,
                                  generations=25,
                                  mutation_probability=mutation_adapter,
                                  crossover_probability=crossover_adapter,
                                  param_grid=param_grid,
                                  n_jobs=-1)

   # Train and optimize the estimator
   evolved_estimator.fit(X_train, y_train)
   # Best parameters found
   print(evolved_estimator.best_params_)
   # Use the model fitted with the best parameters
   y_predict_ga = evolved_estimator.predict(X_test)
   print(accuracy_score(y_test, y_predict_ga))

   # Saved metadata for further analysis
   print("Stats achieved in each generation: ", evolved_estimator.history)
   print("Best k solutions: ", evolved_estimator.hof)

