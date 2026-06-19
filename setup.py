import os
import pathlib
from setuptools import setup, find_packages

# python setup.py sdist bdist_wheel
# twine upload --skip-existing dist/* -u __token__ -p <token>

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
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    project_urls={
        "Documentation": "https://sklearn-genetic-opt.readthedocs.io/en/stable/",
        "Source Code": "https://github.com/rodrigo-arenas/Sklearn-genetic-opt",
        "Bug Tracker": "https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues",
    },
    packages=find_packages(include=["sklearn_genetic", "sklearn_genetic.*"], exclude=["*tests*"]),
    install_requires=[
        "scikit-learn>=1.9.0",
        "numpy>=2.4.6",
        "deap>=1.4.4",
        "tqdm>=4.68.3",
    ],
    extras_require={
        "mlflow": ["mlflow>=3.14.0"],
        "seaborn": ["seaborn>=0.13.2"],
        "tensorflow": [
            "tensorflow>=2.21.0; python_version < '3.14'",
            "tensorboard>=2.20.0,<2.21.0; python_version < '3.14'",
        ],
        "all": [
            "mlflow>=3.14.0",
            "seaborn>=0.13.2",
            "tensorflow>=2.21.0; python_version < '3.14'",
            "tensorboard>=2.20.0,<2.21.0; python_version < '3.14'",
        ],
    },
    python_requires=">=3.12",
    include_package_data=True,
)
