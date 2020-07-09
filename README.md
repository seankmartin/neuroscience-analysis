# PythonTemplate
[![Build Status](https://travis-ci.com/seankmartin/PythonTemplate.svg?branch=master)](https://travis-ci.com/seankmartin/PythonTemplate)
[![Documentation Status](https://readthedocs.org/projects/pythontemplate/badge/?version=latest)](https://pythontemplate.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A template for Python Projects.

## How to convert to your project
1. Rename the folder `your_package` and edit `your_package/__init__.py`
2. Update `setup.py, version.txt, requirements.txt, dev_requirements.txt, LICENSE` to allow for users to install your package using `pip install .` or uploading to PyPI.
3. Update `README.md`.
4. Add the correct Makefile or batch script etc. to docs to allow for building documentation. Most likely this will use sphinx or pdoc3.
5. Add your tests to the tests folder that can be run with pytest or a similar testing framework.

Optionally, you could also setup continuous integration with CircleCI or similar, and setup git hooks.

Check [my website](https://seankmartin.netlify.app/python/getting_your_code_out_there/#uploading-your-package-to-pypi) for more information.

# README Template Below

# Project Name
Give a succinct description of the project.

## Installation
Describe how to install stable version and dev version.

## Dependencies
List what installation requires.

## Documentation
Link to where can further documentation be found.

## Contributing
Show the guide for contributing.

## Licensing
What License the project is provided under.
