.. sklearn-genetic-opt documentation master file, created by
   sphinx-quickstart on Sat May 29 19:27:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sklearn-genetic-opt
==================
scikit-learn models hyperparameters tuning and feature selection,
using evolutionary algorithms.

#################################################################

This is meant to be an alternative to popular methods inside scikit-learn such as Grid Search and Randomized Grid Search
for hyperparameters tuning, and from RFE, Select From Model for feature selection.

Sklearn-genetic-opt uses evolutionary algorithms from the deap package to choose a set of hyperparameters
that optimizes (max or min) the cross-validation scores, it can be used for both regression and classification problems.

Installation:
#############

Install sklearn-genetic-opt

It's advised to install sklearn-genetic using a virtual env, to install a light version,
inside the env use::

   pip install sklearn-genetic-opt

.. |PythonMinVersion| replace:: 3.7
.. |ScikitLearnMinVersion| replace:: 0.21.3
.. |NumPyMinVersion| replace:: 1.14.5
.. |SeabornMinVersion| replace:: 0.9.0
.. |DEAPMinVersion| replace:: 1.3.1
.. |MLflowMinVersion| replace:: 1.17.0
.. |TensorflowMinVersion| replace:: 2.0.0
.. |tqdmMinVersion| replace:: 4.61.1

sklearn-genetic-opt requires:

- Python (>= |PythonMinVersion|)
- scikit-learn (>= |ScikitLearnMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- DEAP (>= |DEAPMinVersion|)
- tqdm (>= |tqdmMinVersion|)

Extra requirements:

These requirements are necessary to use
:mod:`~sklearn_genetic.plots`, :class:`~sklearn_genetic.mlflow.MLflowConfig`
and :class:`~sklearn_genetic.callbacks.TensorBoard` correspondingly.

- Seaborn (>= |SeabornMinVersion|)
- MLflow (>= |MLflowMinVersion|)
- Tensorflow (>= |TensorflowMinVersion|)

This command will install all the extra requirements, except for Tensorflow,
as it is usually advised to look further which distribution works better for you::

   pip install sklearn-genetic-opt[all]

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: User Guide / Tutorials:

   tutorials/basic_usage
   tutorials/callbacks
   tutorials/custom_callback
   tutorials/understand_cv
   tutorials/mlflow

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Jupyter notebooks examples:

   notebooks/sklearn_comparison.ipynb
   notebooks/Boston_Houses_decision_tree.ipynb
   notebooks/Iris_feature_selection.ipynb
   notebooks/Digits_decision_tree.ipynb
   notebooks/MLflow_logger.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Release Notes

   release_notes

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/gasearchcv
   api/featureselectioncv
   api/callbacks
   api/plots
   api/mlflow
   api/space
   api/algorithms


.. toctree::
   :maxdepth: 1
   :caption: External References:

   external_references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

