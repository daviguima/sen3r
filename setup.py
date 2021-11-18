from setuptools import setup, find_packages
import sen3r
import os

README = os.path.join(os.path.dirname(__file__), 'README.md')

setup(name='sen3r',
      version=sen3r.__version__,
      description='SEN3R (Sentinel-3 Reflectance Retrieval over Rivers) enables extraction of reflectance time series from images over water bodies.',
      long_description=open(README).read(),
      long_description_content_type='text/markdown',
      author="David Guimarães",
      author_email="dvdgmf@gmail.com",
      license="MIT",
      platforms='any',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
      ],
      packages=find_packages(),
      python_requires='>=3.6',
      entry_points={
          'console_scripts': ['sen3r = main:main']
      },
      include_package_data=True,
      package_data={'sen3r': ['../main.py']},
      keywords='Sentinel-3 OLCI WFR SYN Relfectance CAMS time-series',
      url='http://github.com/hybam-dev/sen3r/', )
