# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='slib',
    version='0.0.1',
    description='Python library for manageing data science experiments',
    long_description=readme,
    author='Tillmann Radmer',
    author_email='tillmann.radmer@gmail.com',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

