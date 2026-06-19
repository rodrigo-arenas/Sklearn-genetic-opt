# Contributing

Sklearn-genetic-opt is an open-source project and contributions of all kinds
are welcome.

You can contribute with documentation, examples/tutorials, reviewing pull requests, code,
helping answer questions in issues, creating visualizations, maintaining project
infrastructure, and creating new tests. 

Code contributions are always welcome, from simple bug fixes to new features.
Also, consider contributing to the documentation, 
and reviewing open issues, it is the easiest way to get started.

## Local development setup

sklearn-genetic-opt supports Python 3.12 and newer. Python 3.13 is a good
default for local development because it can run the regular test suite and the
TensorFlow/TensorBoard optional tests.

Create and activate a virtual environment, then install the project in editable
mode with the development dependencies:

```bash
python -m pip install --upgrade pip
pip install -r dev-requirements.txt
```

The `dev-requirements.txt` file is now a small wrapper around the dependency
groups declared in `pyproject.toml`. You can also install only the dependency
groups you need:

```bash
# Core package only
pip install -e .

# Tests
pip install -e ".[test,plot,mlflow]"

# Tests with TensorBoard support
pip install -e ".[test,plot,mlflow,tensorflow]"

# Documentation
pip install -e ".[docs]"

# Packaging checks
pip install -e ".[build]"

# Formatting
pip install -e ".[lint]"
```

## Useful local commands

Check formatting:

```bash
black --check .
```

Format the code:

```bash
black .
```

Run the test suite without coverage:

```bash
pytest sklearn_genetic/
```

Run the coverage check used by CI:

```bash
pytest sklearn_genetic/ --cov-fail-under=95 --cov=./ --cov-report=term-missing:skip-covered
```

Build the package and check the generated distribution metadata:

```bash
python -m build
twine check dist/*
```

Build the documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

If the documentation build processes notebooks, make sure `pandoc` is installed
on your system.

Run a quick fit-performance benchmark smoke check:

```bash
python benchmarks/benchmark_fit.py --quick
```

For performance-sensitive changes, compare against a saved baseline:

```bash
python benchmarks/benchmark_fit.py --label baseline --output-json benchmarks/baseline.json
python benchmarks/benchmark_fit.py --label current --compare-json benchmarks/baseline.json
```

The benchmark reports wall time, cross-validation call counts, cache/duplicate
evaluation counters, optimizer telemetry such as diversity and stagnation, and
model metrics. Prefer using the same machine, Python environment, random seed,
and benchmark options when comparing results.

If you have questions, you can open an issue (tag it as a question).

We encourage you to follow these guidelines:

* Fork this project, make the changes you expect to merge and make a pull request 
* If the work you are making is related to some issue, please mention in the comments 
  that you are working on it, so other people know and no duplicate your work.
* If you are working on a new feature, or have an idea, consider first opening an issue
  so people know what you are working on and possibly give some guidelines
* Commit all changes by pull request (PR)
* A PR solves one problem (do not mix problems in one PR) with the
  minimal set of changes
* The changes should come with their respective tests and documentation
* Describe why you are proposing those changes 
* Please run Black to keep the formatting style.
* Make sure all the tests are passing by running `pytest sklearn_genetic/` in the root of the project.
* We can not merge if the tests fail.

# External References

We have a [file](https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/docs/external_references.rst) 
in our docs where you can put external references with the use of sklearn-genetic-opt,
it could be a blog post, a video or an article.
You can add the link of the content in that file (following the contribution guides).

Take into consideration:

* The link must be after the last existing link.
* The name must be the title that will be visible.
* The link must take to the referred content.

## Thank you for being part of this project!
