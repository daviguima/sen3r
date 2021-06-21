# coding: utf-8
from setuptools import setup
import os


README = os.path.join(os.path.dirname(__file__), 'README.md')

setup(name='sen3r',
      version='1.0',
      description='SEN3R (Sentinel-3 Reflectance Retrieval over Rivers) enables extraction of reflectance time series from images over water bodies.',
      long_description=open(README).read(),
      long_description_content_type='text/markdown',
      author="David Guimarães", author_email="dvdgmf@gmail.com",
      license="MIT",
      py_modules=['decouple'],
      zip_safe=False,
      platforms='any',
      include_package_data=True,
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
      ],
      url='http://github.com/hybam-dev/sen3r/',)