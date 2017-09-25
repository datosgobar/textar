#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

with open("requirements.txt") as f:
    requirements = [req.strip() for req in f.readlines()]

with open("requirements_dev.txt") as f:
    test_requirements = [req.strip() for req in f.readlines()]

setup(
    name='textar',
    version='0.0.6',
    description="Paquete en python para análisis, clasificación y recuperación de textos, utilizado por el equipo de Datos Argentina.",
    long_description=readme + '\n\n' + history,
    author="Datos Argentina",
    author_email='datos@modernizacion.gob.ar',
    url='https://github.com/datosgobar/textar',
    packages=[
        'textar',
    ],
    package_dir={'textar':
                 'textar'},
    include_package_data=True,
    package_data={"textar": ["*.txt"]},
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='textar',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Spanish',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
