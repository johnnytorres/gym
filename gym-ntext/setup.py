from setuptools import find_packages
from setuptools import setup

#REQUIRED_PACKAGES = ['tensorflow>=1.7', 'gym']
REQUIRED_PACKAGES = ['tensorflow', 'keras', 'sklearn', 'gym']

setup(
    name='gym-ntext',
    version='0.0.3',
    install_requires=REQUIRED_PACKAGES,
    packages=['agents'],#find_packages(),
    include_package_data=True,
    description=''
)
