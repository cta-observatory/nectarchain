#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# import sys

from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='nectarchain',
    packages=find_packages(),
    version='0.1',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'astropy',
        'ctapipe',
        'protozfits @ https://github.com/cta-sst-1m/protozfitsreader/archive/v1.5.0.tar.gz',
    ],
    tests_require=['pytest'],
    setup_requires=['pytest_runner'],
    author='NectarCAM collaboration',
    url='https://github.com/cta-observatory/nectarchain',
    license=''
)

