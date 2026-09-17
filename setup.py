import sys

from setuptools import Extension, setup

# The C kernels are optional: if compilation fails, fincurve falls back to NumPy.
setup(
    ext_modules=[
        Extension(
            "fincurve._kernels",
            sources=["fincurve/_kernels.c"],
            extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
            optional=True,
        )
    ]
)
