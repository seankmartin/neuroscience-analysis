
PythonTemplate
==============

A template for Python Projects.

How to convert to your project
------------------------------


#. Rename the folder ``your_package`` and edit ``your_package/__init__.py``
#. Update ``setup.py, version.txt, requirements.txt, dev_requirements.txt, LICENSE`` to allow for users to install your package using ``pip install .`` or uploading to PyPI.
#. Update ``README.md``.
#. Add the correct Makefile or batch script etc. to docs to allow for building documentation. Most likely this will use sphinx or pdoc3.
#. Add your tests to the tests folder that can be run with pytest or a similar testing framework.

Optionally, you could also setup continuous integration with CircleCI or similar, and setup git hooks.

Checkout TODO put link for a guide to getting a project like this formatted, linted, and shared on PyPI and Read the Docs.

README Template Below
=====================

Project Name
============

Give a succinct description of the project.

Installation
------------

Describe how to install stable version and dev version.

Dependencies
------------

List what installation requires.

Documentation
-------------

Link to where can further documentation be found.

Contributing
------------

Show the guide for contributing.

Licensing
---------

What License the project is provided under.
