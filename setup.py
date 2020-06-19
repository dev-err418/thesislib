from setuptools import setup

setup(
   name='thesislib',
   version='1.0',
   description='Collection of functions/utilities for my thesis',
   author='O.S. Agba',
   author_email='',
   packages=['thesislib'],
   install_requires=['pandas', 'numpy', 'matplotlib', 'requests', 'tabulate', 'scikit-learn', 'dateutil'],
)