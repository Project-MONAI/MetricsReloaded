#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
from glob import glob

from pkg_resources import parse_requirements
from setuptools import find_packages
from setuptools import setup

source_dir = os.path.abspath(os.path.dirname(__file__))

# read the version and other strings from _version.py
version_info = {}
with open(os.path.join(source_dir, "MetricsReloaded/_version.py")) as o:
    exec(o.read(), version_info)

# read install requirements from requirements.txt
with open(os.path.join(source_dir, "requirements.txt")) as o:
    requirements = [str(r) for r in parse_requirements(o.read())]

setup(
    name='MetricsReloaded',
    version=version_info['__version__'],
    description=version_info['__description__'],
    author=version_info['__author__'],
    author_email=version_info['__author_email__'],
    url='https://github.com/csudre/MetricsReloaded',
    packages=find_packages(),
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob('MetricsReloaded/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering'
    ],
    project_urls={
        'Documentation': 'https://MetricsReloaded.readthedocs.io/',
        'Issue Tracker': 'https://github.com/csudre/MetricsReloaded/issues',
    },
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
    },
    setup_requires=[
        'pytest-runner',
    ],
    entry_points={
    },
)
