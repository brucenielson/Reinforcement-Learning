# https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
# https://www.jetbrains.com/help/pycharm/cython.html#get-started-cython
# How to run Cython: Tools->Run setup.py -> built_ext -> --inplace

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
# noinspection PyPackageRequirements
from Cython.Build import cythonize

# noinspection PyPackageRequirements
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize(["q_learner_interfaces.py", "q_table.py", "q_learner.py", "open_gym_taxi.py",
                           "environments.py"], annotate=True, language_level=3), include_dirs=[numpy.get_include()]
)
