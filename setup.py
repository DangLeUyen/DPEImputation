from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='DPEImputation',
    version='0.1.0',
    description='My awesome DPEImputation package',
    url='https://github.com/DangLeUyen/DPEImputation.git',
    package_dir={"": "src"},
    install_requires=requirements,
    keywords='python package',
    python_requires='>=3.6',
)