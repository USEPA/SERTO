import os
import ast
from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))


def get_version(module='serto'):
    """Get version."""

    version = '0.0.0.dev0'

    with open(os.path.join(HERE, module, '__init__.py'), 'r') as f:
        data = f.read()
    lines = data.split('\n')
    for line in lines:
        if line.startswith('VERSION_INFO'):
            version_tuple = ast.literal_eval(line.split('=')[-1].strip())
            version = '.'.join(map(str, version_tuple))
            break

    return version


setup(
    name='serto',
    version=get_version(),
    author="See AUTHORS",
    packages=['serto'],
    entry_points={
        'console_scripts': [
            'serto = serto.cli:main',
        ]
    },
)
