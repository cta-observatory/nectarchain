#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# import sys
import setuptools


setuptools.setup(name='nectarchain',
                 version=0.1,
                 description="DESCRIPTION",  # these should be minimum list of what is needed to run
                 packages=setuptools.find_packages(),
                 install_requires=['h5py',
                                   'seaborn'
                                   ],
                 tests_require=['pytest', 'pytest-ordering'],
                 author='NectarCAM collaboration',
                 author_email='',
                 license='',
                 url='https://github.com/cta-observatory/nectarchain',
                 long_description='',
                 classifiers=[],
                 )
