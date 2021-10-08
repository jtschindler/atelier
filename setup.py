#!/usr/bin/env python

from distutils.core import setup
import pathlib

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

setup(name='atelier',
      version='0.1a0',
      description='The "Atelier" is a compilation of software and tools designed for my astrophysical research.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan-Torge Schindler',
      author_email='schindler@mpia.de',
      license='BSD-3',
      url='http://github.com/jtschindler/atelier',
      packages=['atelier'],
      provides=['atelier'],
      package_dir={'atelier': 'atelier'},
      # package_data={'atelier': ['data//*.*']}
      # scripts=['atelier/'],
      keywords=['Atelier', 'astronomy', 'spectroscopy', 'modeling', 'fitting'],
      classifiers = ['Development Status :: 2 - Alpha',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3.9',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Software Development :: User Interfaces',
                   'Topic :: Software Development :: Libraries :: Python Modules'
                   ]
    )
