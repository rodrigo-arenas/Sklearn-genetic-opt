import os
import pathlib
from setuptools import setup, find_packages

# python setup.py sdist bdist_wheel
# twine upload --skip-existing dist/*

# get __version__ from _version.py
ver_file = os.path.join("sklearn_genetic", "_version.py")
with open(ver_file) as f:
    exec(f.read())

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.rst").read_text()
setup(
    name="sklearn-genetic-opt",
    version=__version__,
    description="Scikit-learn models hyperparameters tuning and features selection, using evolutionary algorithms",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/rodrigo-arenas/Sklearn-genetic-opt",
    author="Rodrigo Arenas",
    author_email="rodrigo.arenas456@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    project_urls={
        "Documentation": "https://sklearn-genetic-opt.readthedocs.io/en/stable/",
        "Source Code": "https://github.com/rodrigo-arenas/Sklearn-genetic-opt",
        "Bug Tracker": "https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues",
    },
    packages=find_packages(
        include=["sklearn_genetic", "sklearn_genetic.*"], exclude=["*tests*"]
    ),
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.19.0",
        "deap>=1.3.3",
        "tqdm>=4.61.1",
    ],
    extras_require={
        "mlflow": ["mlflow>=1.17.0"],
        "seaborn": ["seaborn>=0.11.2"],
        "tensorflow": ["tensorflow>=2.0.0"],
        "all": ["mlflow>=1.30.0", "seaborn>=0.11.2", "tensorflow>=2.0.0"],
    },
    python_requires=">=3.9",
    include_package_data=True,
)
