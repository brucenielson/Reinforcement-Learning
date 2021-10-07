# from setuptools import setup, Extension
#
# module = Extension('Taxi', sources=['open_gym_taxi.py', 'q_learner.py'])
#
# setup(
#     name='cythonTest',
#     version='1.0',
#     author='jetbrains',
#     ext_modules=[module]
# )

from distutils.core import setup

import numpy
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize(["q_learner_interfaces.py", "q_table.py", "q_learner.py", "open_gym_taxi.py", "environments.py"], annotate=True, language_level=3), include_dirs=[numpy.get_include()]
)
