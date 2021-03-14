#! /usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

setuptools.setup(
    name="pyro-util",
    version="0.0.1",
    description="Utilities for pytorch + pyro projects",
    url="www.github.com/jamestwebber/pyro-util",
    author="James Webber",
    author_email="jwebber@broadinstitute.org",
    license="MIT",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=["torch", "pyro-ppl", "scikit-learn"],
    extras_require={
        "dev": ["pre-commit", "pytest", "black", "isort", "flake8"],
        "cupy": ["cupy"],
    },
    zip_safe=False,
)
