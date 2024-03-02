# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

install_requires = [
]

setup(
    name="cts",
    version="0.1.0",
    description="Categorical Time Series Analysis with Sequential Pattern Mining",
    author="Alexander Grote",
    author_email="grote@fzi.de",
    url="",
    packages=find_packages(exclude=("tests", "docs")),
)