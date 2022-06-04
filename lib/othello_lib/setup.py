from setuptools import setup
from setuptools import setup, find_packages

setup(
    name = "othello_lib",
    version="0.0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
