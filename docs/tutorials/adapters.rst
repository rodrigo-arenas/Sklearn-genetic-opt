Using Adapters
==============

Introduction
------------

You can define adapters to have a dynamic mutation and crossover probabilities over the optimization
instead of a fixed value. The idea is to make these probabilities a function of the generations;
this definition can enable different training strategies, for example:

* Start with a high probability mutation to explore more diverse solutions and slowly reduce it
  to stay with the more promising ones.
* Start with a low crossover and end with a higher probability
* Combine different strategies for each parameter

All the methods uses three parameters:

* **initial_value:** This is the value used at generation 0
* **end_value:** It's the limit value that the parameter can take, starting from initial_value
* **adaptive_rate**: Controls how fast the value approaches the end_value;
  greater values increase the speed of convergence

For the following sections, it's important to understand this notation:

===================== ===============
Name                  Notation
===================== ===============
initial_value         :math:`p_0`
end_value             :math:`p_f`
current generation    :math:`t`
adaptive_rate         :math:`\alpha`
value at generation t :math:`p(t; \alpha)`
===================== ===============

Note that :math:`p_0` doesn't need to be greater than :math:`p_f`.

If :math:`p_0 > p_f`, you are performing a decay towards :math:`p_0`.

If :math:`p_0 < p_f`, you are performing an ascend towards :math:`p_f`.

All the adapters :math:`p(t; \alpha)`, for :math:`\alpha \in (0,1)`,
have the following properties:

.. math::

   \lim_{t->0^{+}} p(t; \alpha) = p_0\\
   \\
   \lim_{t->\infty} p(t; \alpha) = p_f

The following adapters are available:

* ExponentialAdapter
* InverseAdapter
* PotentialAdapter

ExponentialAdapter
------------------

The Exponential Adapter uses the following form to change the initial value

.. math::

   p(t; \alpha) = (p_0-p_f)e^{-\alpha t} + p_f

Usage example:

.. code:: python3

    from sklearn_genetic.schedules import ExponentialAdapter

    # Decay over initial_value
    adapter = ExponentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)

    # Run a few iterations
    for _ in range(3):
        adapter.step()  # 0.8, 0.74, 0.69

This is how the adapter looks for different values of alpha

**decay:**

.. image:: ../images/schedules_exponential_0.png

**ascend:**

.. image:: ../images/schedules_exponential_1.png

.. code:: python3

   import matplotlib.pyplot as plt
   from sklearn_genetic.schedules import ExponentialAdapter

   values = [{"initial_value": 0.8, "end_value": 0.2},  # Decay
             {"initial_value": 0.2, "end_value": 0.8}]  # Ascend
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

The Inverse Adapter uses the following form to change the initial value

.. math::

   p(t; \alpha) = \frac{(p_0-p_f)}{1+\alpha t} + p_f

Usage example:

.. code:: python3

    from sklearn_genetic.schedules import InverseAdapter

    # Decay over initial_value
    adapter = InverseAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)

    # Run a few iterations
    for _ in range(3):
        adapter.step()  # 0.8, 0.75, 0.7

This is how the adapter looks for different values of alpha

**decay:**

.. image:: ../images/schedules_inverse_0.png

**ascend:**

.. image:: ../images/schedules_inverse_1.png

PotentialAdapter
----------------

The Inverse Adapter uses the following form to change the initial value

.. math::

   p(t; \alpha) = (p_0-p_f)(1-\alpha)^{ t} + p_f

Usage example:

.. code:: python3

    from sklearn_genetic.schedules import PotentialAdapter

    # Decay over initial_value
    adapter = PotentialAdapter(initial_value=0.8, end_value=0.2, adaptive_rate=0.1)

    # Run a few iterations
    for _ in range(3):
        adapter.step()  # 0.8, 0.26, 0.206

This is how the adapter looks for different values of alpha

**decay:**

.. image:: ../images/schedules_potential_0.png

**ascend:**

.. image:: ../images/schedules_potential_1.png

Compare
-------

This is how all adapters looks like for the same value of alpha

**decay:**

.. image:: ../images/schedules_comparison_0.png

**ascend:**

.. image:: ../images/schedules_comparison_0.png

.. code:: python3

   import matplotlib.pyplot as plt
   from sklearn_genetic.schedules import ExponentialAdapter, PotentialAdapter, InverseAdapter


   params = {"initial_value": 0.2, "end_value": 0.8, "adaptive_rate": 0.15}  # Ascend
   adapters = [ExponentialAdapter(**params), PotentialAdapter(**params), InverseAdapter(**params)]

   for adapter in adapters:
       adapter_result = [adapter.step() for _ in range(50)]

       plt.plot(adapter_result, label=f"{type(adapter).__name__}")

   plt.xlabel(r'$t$')
   plt.ylabel(r'$p(t; \alpha)$')
   plt.title("Adapters Comparison")
   plt.legend()
   plt.show()


