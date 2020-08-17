from setuptools import setup, find_packages

setup(
   name='thesislib',
   version='1.1.4',
   description='Collection of functions/utilities for my thesis',
   author='O.S. Agba',
   author_email='',
   packages=find_packages(),
   install_requires=['pandas', 'numpy', 'matplotlib', 'requests', 'tabulate', 'scikit-learn', 'python-dateutil', 'torch'
                     'visdom', 'boto3'],
)