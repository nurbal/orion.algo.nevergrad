#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for `orion.algo.nevergrad`."""
import os

from setuptools import setup

import versioneer

import nevergrad as ng

repo_root = os.path.dirname(os.path.abspath(__file__))

tests_require = ["pytest>=3.0.0"]

setup_args = dict(
    name="orion.algo.nevergrad",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="TODO",
    long_description=open(
        os.path.join(repo_root, "README.rst"), "rt", encoding="utf8"
    ).read(),
    license="BSD-3-Clause",
    author="Epistimio",
    author_email="xavier.bouthillier@mila.quebec",
    url="https://github.com/Epistimio/orion.algo.nevergrad",
    packages=["orion.algo.nevergrad"],
    package_dir={"": "src"},
    include_package_data=True,
    entry_points={
        "BaseAlgorithm": [],
    },
    install_requires=["orion>=0.1.15", "numpy", "nevergrad"],
    tests_require=tests_require,
    setup_requires=["setuptools", "pytest-runner>=2.0,<3dev"],
    extras_require=dict(test=tests_require),
    # "Zipped eggs don't play nicely with namespace packaging"
    # from https://github.com/pypa/sample-namespace-packages
    zip_safe=False,
)

# Dynamically create entry points using nevergrad optimizers registry
for algo_name in ng.optimizers.registry.keys():
    setup_args["entry_points"]["BaseAlgorithm"].append(
        'nevergrad_{name} = orion.algo.nevergrad.nevergradoptimizer:Nevergrad_{name}'.format(namelow=algo_name.lower(),name=algo_name)
    )

setup_args["keywords"] = [
    "Machine Learning",
    "Deep Learning",
    "Distributed",
    "Optimization",
]

setup_args["platforms"] = ["Linux"]

setup_args["classifiers"] = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
] + [("Programming Language :: Python :: %s" % x) for x in "3 3.6 3.7 3.8 3.9".split()]

if __name__ == "__main__":
    setup(**setup_args)
