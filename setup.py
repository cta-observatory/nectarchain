#!/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from setuptools import setup, find_packages

setup(name='nectarchain',
      description='High level analysis tools for NectarCAM',
      url='',
      author='NectarCAM collaboration',
      author_email='',
      license='BSD',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=[
          'protozfits',
          'ctapipe',
          'ctapipe_io_nectarcam',
      ],
      tests_require=[
          'pytest',
      ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Development Status :: 1 - Planning',
      ],
      zip_safe=False,
      use_2to3=False
)
