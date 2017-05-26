#!/usr/bin/env python

import os
from setuptools import setup

setup(name='l3s-dbce',
      version="0.0.1dev",
      license="MIT License",
      description='Command line tools and libraries for diff based content extraction from WARC files',
      author='Rafal Lesniak',
      author_email='lesniak@dcsec.uni-hannover.de',
      packages=['dbce', 'dbce/datatypes'],
      scripts=['dbce-tool.py'],
    )
