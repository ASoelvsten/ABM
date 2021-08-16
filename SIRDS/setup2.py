from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension("SIRDS",
              sources=["SIRD.pyx"],
              libraries=["m"],  # Unix-like specific
              include_dirs = [numpy.get_include()],
              extra_compile_args=["-O3"], 
              language="c++"
              )
]

setup(
    ext_modules = cythonize(ext_modules),
)
