from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(
    name='bankup',
    version='0.0.1',
    description='package from project which aims is to manage budget',
    url='git@github.com:arnaudczls/bankup.git',
    author='Arnaud CAZELLES',
    author_email='a.cazelles@gmail.com',
    license='unlicense',
    packages=find_packages(),
    install_requires=requirements,
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False
)