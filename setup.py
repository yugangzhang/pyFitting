"""
pyFitting - Modular Fitting Framework
Setup script
"""

#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import sys
import warnings
import versioneer

from setuptools import setup, find_packages
 

 
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyFitting",
    version="0.1.0",
    #version=versioneer.get_version(),
    author='Brookhaven National Laboratory_YugangZhang@CFN',
    author_email="yuzhang@bnl.gov",
    description="A clean, modular, extensible framework for curve fitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yugangzhang/pyFitting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "matplotlib>=3.3.0",
        ],
    },
)