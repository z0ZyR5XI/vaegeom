from setuptools import setup, find_packages

setup(
    name='vaegeom',
    version='1.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch==1.12.1'
    ]
)
