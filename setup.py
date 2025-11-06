from __future__ import annotations
from setuptools import setup, Extension, find_packages
import numpy

setup(
    packages=find_packages(),
    ext_modules=[
        Extension(
            "pycramer._lookup",
            sources=["pycramer/_lookup.cpp"],
            include_dirs=[numpy.get_include()],
            language="c++",
        ),
    ],
)
