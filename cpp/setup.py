"""Build the C++ Monte Carlo module (works with MSVC on Windows and g++/clang).

Run from the cpp/ folder:
    python setup.py build_ext --build-lib ../src

The compiled module is written to src/riskengine/ (a .pyd file on Windows,
a .so file on Linux/macOS), next to the Python package, so that
`from riskengine import mc_engine` works.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "riskengine.mc_engine",      # the package it lives in + the module name
    ["mc_engine.cpp"],
    cxx_std=17,
)

setup(
    name="riskengine-mc-engine",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)