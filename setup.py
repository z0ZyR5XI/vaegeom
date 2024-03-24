from setuptools import setup, find_packages

setup(
    name='vaegeom',
    version='1.2.0',
    packages=find_packages(),
    install_requires=[
        'lightning',
        'torch'
    ]
)
