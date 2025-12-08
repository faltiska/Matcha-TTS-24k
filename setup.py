import os
import numpy
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

exts = [
    Extension(
        name="matcha.utils.monotonic_align.core",
        sources=["matcha/utils/monotonic_align/core.pyx"],
        extra_compile_args=[
            '-O3',           # Maximum optimization
            '-march=native', # Optimize for your CPU architecture
            '-ffast-math',   # Fast floating point math
        ],
        include_dirs=[numpy.get_include()],
    )
]

with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "matcha", "VERSION"), encoding="utf-8") as fin:
    version = fin.read().strip()


def get_requires():
    requirements = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements, encoding="utf-8") as reqfile:
        return [str(r).strip() for r in reqfile]


setup(
    name="matcha-tts",
    version=version,
    description="🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Shivam Mehta",
    author_email="shivam.mehta25@gmail.com",
    url="https://shivammehta25.github.io/Matcha-TTS",
    install_requires=get_requires(),
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),
    entry_points={
        "console_scripts": [
            "matcha-data-stats=matcha.utils.generate_data_statistics:main",
            "matcha-tts=matcha.cli:cli",
            "matcha-tts-app=matcha.app:main",
            "matcha-tts-get-durations=matcha.utils.get_durations_from_trained_model:main",
        ]
    },
    ext_modules=cythonize(
        exts,
        language_level=3,
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,
        }
    ),
    python_requires=">=3.9.0",
)