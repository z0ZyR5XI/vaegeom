from setuptools import setup, find_packages

setup(
    name='vaegeom',
    version='1.2.1',
    packages=find_packages(),
    install_requires=[
        'lightning',
        'torch'
    ]
)
