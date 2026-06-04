"""Setup script for building the SympleX Python extension.

Build with:
    cd python && pip install -e .

Or manually:
    cd python && python setup.py build_ext --inplace
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    """Custom build_ext that uses CMake to build the pybind11 module."""

    def build_extension(self, ext):
        # The extension is built by CMake, not by setuptools directly.
        # We just invoke cmake and copy the result.
        extdir = Path(os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))))

        # Source directory (SympleX root)
        src_dir = Path(__file__).resolve().parent.parent

        # Build directory
        build_dir = Path(os.path.join(os.path.abspath(self.build_temp), "cmake_build"))
        build_dir.mkdir(parents=True, exist_ok=True)

        # CMake configure
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

        try:
            import pybind11
            cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        except ImportError:
            pass

        subprocess.check_call(
            ["cmake", str(src_dir)] + cmake_args, cwd=build_dir
        )

        # Build just the _symplex target
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_symplex", "-j"],
            cwd=build_dir,
        )


# A dummy extension to trigger CMakeBuild
ext = Extension(
    "symplex._symplex",
    sources=[],  # CMake handles the actual compilation
)

setup(
    name="simpleX",
    version="1.0.0",
    packages=["symplex"],
    ext_modules=[ext],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.10",
)
