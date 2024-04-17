from setuptools import setup, find_packages

setup(
    name='czone',
    version='2022.09.20',
    description='An open source python package for generating nanoscale+ atomic scenes',
    url='https://github.com/lerandc/construction_zone',
    author='Luis Rangel DaCosta',
    author_email='luisrd@berkeley.edu',
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=[
        'pymatgen',
        'numpy',
        'scipy',
        'ase',
        'wulffpack']
)
