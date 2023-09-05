#!/usr/bin/env python
# coding=utf-8
"""The setup script."""

import os
from setuptools import find_packages, setup  # type: ignore

with open("README.md") as readme_file:
    long_description = readme_file.read()

install_requires = [
    "numpy==1.24.2",
    "torch==2.0.1", 
    "pytorch-lightning==2.0.2",
    "torchvision==0.15.2",
    "pydantic==1.10.8",
    "tensorboard>=2.9",
    "mlflow>=1.26",
    "hydra-core==1.3.0",
    "python-dotenv>=0.20",
    "tqdm==4.64",
    "rich>=12.4",
    "hydra-submitit-launcher==1.2.0",
    "hydra-optuna-sweeper==1.2.0",
    "hydra-colorlog==1.2.0",
    "matplotlib>=3.5.3",
    "seaborn>=0.11.2",
    "pandas>=1.4.1",
    "scikit-learn>=1.1.2",
    "x-transformers==1.19.0"
]


setup(
    author="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    description="STAPLER",
    install_requires=install_requires,
    extras_require={
        "dev": ["pytest", "numpydoc", "pylint", "black==22.3.0"],
    },
    license="",
    include_package_data=True,
    name="stapler",
    test_suite="tests",
    url="https://github.com/NKI-AI/STAPLER",
    py_modules=["stapler"]
    # version=version,
    # zip_safe=False,
)
