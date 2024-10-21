import sys
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

if sys.platform == "win32":
    cpp_flags = ["/std:c++latest", "/O2"]
else:
    cpp_flags = ["-std=c++23", "-O2"]

extensions = cythonize(
    [
        Extension(
            "biperscan._impl",
            ["src/biperscan/_impl.pyx"],
            extra_compile_args=cpp_flags,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[numpy.get_include()],
            language="c++",
        )
    ],
    language_level=3,
)

setup(ext_modules=extensions)
