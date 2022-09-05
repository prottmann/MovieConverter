#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='converter',
    version='0.1',
    description='Tool for Converting movie to specific format',
    long_description=long_description,
    author='Peter Rottmann',
    author_email='peter.rottmann@uni-bonn.de',
    url='https://twitter.com/Pero_95',
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=required,
    scripts=["bin/converter", "bin/mergeTrp"],
    license='MIT',
)
