from __future__ import annotations

from setuptools import Extension, setup
import numpy


extensions = [
    Extension(
        "pycramer._lookup",
        sources=["pycramer/_lookup.cpp"],
        include_dirs=[numpy.get_include()],
    )
]


setup(ext_modules=extensions)
