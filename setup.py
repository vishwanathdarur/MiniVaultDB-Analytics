import setuptools
from setuptools import Extension

import pybind11

ext_modules = [
    Extension(
        "minivaultdb._native",
        sources=[
            "minivaultdb/_native.cpp",
            "/home/vishwa/Project/MiniVaultDB/src/db/db.cpp",
            "/home/vishwa/Project/MiniVaultDB/src/engine/memtable.cpp",
            "/home/vishwa/Project/MiniVaultDB/src/engine/sstable.cpp",
            "/home/vishwa/Project/MiniVaultDB/src/engine/wal.cpp",
            "/home/vishwa/Project/MiniVaultDB/src/util/hash.cpp",
            "/home/vishwa/Project/MiniVaultDB/src/util/crc32.cpp",
            "/home/vishwa/Project/MiniVaultDB/src/util/arena.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            "/home/vishwa/Project/MiniVaultDB/include",
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O2"],
    )
]

setuptools.setup(
    name="minivaultdb",
    version="0.1.0",
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    zip_safe=False,
)
