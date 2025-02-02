from setuptools import setup, find_packages

setup(
    name="vectorfitting",
    version="0.1.0",
    author="Milan Rother",
    author_email="milan.rother@gmx.de",
    description="Python implementation of the Fast-Relaxed-Vectorfitting algorithm.",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy"
    ],
)