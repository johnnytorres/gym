from setuptools import setup, find_packages

REQUIRED_PACKAGES = [ 'gym']

setup(
    name='gym-ntext-env',
    version='0.0.4',
    install_requires=REQUIRED_PACKAGES,
    packages=['ntext', 'ntext.envs', 'ntext.envs.datasets'], #find_packages(),
)
