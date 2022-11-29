#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
        'astropy~=4.2',
        'ctapipe~=0.12',
        'numpy~=1.22.4',
    ],
    tests_require=['pytest'],
    setup_requires=['pytest_runner'],
    author='NectarCAM collaboration',
    url='https://github.com/cta-observatory/nectarchain',
    license=''
)

