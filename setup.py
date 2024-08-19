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
    #     classifiers=[
    #         'Development Status :: 3 - Alpha',
    #         'Intended Audience :: Developers',
    #         'Topic :: Software Development :: Libraries',
    #         'License :: OSI Approved :: MIT License',
    #         'Programming Language :: Python :: 3',
    #         'Programming Language :: Python :: 3.6',
    #         'Programming Language :: Python :: 3.7',
    #         'Programming Language :: Python :: 3.8',
    #         'Programming Language :: Python :: 3.9',
    #     ],
    keywords='python package',
    python_requires='>=3.6',
)