import pathlib
from setuptools import setup, find_packages

# python setup.py sdist bdist_wheel
# twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/*

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="sklearn-genetic-opt",
    version="0.2.0",
    description="Sklearn models hyperparameters tuning using genetic algorithms",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rodrigo-arenas/Sklearn-genetic-opt",
    author="Rodrigo Arenas",
    author_email="rodrigo.arenas456@gmail.com",
    license="MIT",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",

    ],
    packages=find_packages(include=['sklearn_genetic', 'sklearn_genetic.*']),
    install_requires=['scikit-learn>=0.21.3', 'numpy>=1.13.3', 'seaborn>=0.11.1', 'deap>=1.3.1'],
    python_requires=">=3.6",
    include_package_data=True,
)
