from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(
    name='bankup',
    version='0.0.3',
    description='package from project which aims is to manage budget',
    url='git@github.com:arnaudczls/bankup.git',
    author='Arnaud CAZELLES',
    author_email='a.cazelles@gmail.com',
    license='unlicense',
    packages=find_packages(),
    #Import les models ML.
    #Dans le package bankup il y a le repertoire model qui contient tous les models
    package_data={'bankup': ['*.json','model/*.joblib']},
    install_requires=requirements,
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False
)