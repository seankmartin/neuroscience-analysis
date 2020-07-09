import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def readlines(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).readlines()


def parse_version(fname="version.txt"):
    version_info = readlines(fname)
    version_dict = {}
    for line in version_info:
        key, val = line.strip().split("=")
        key, val = key.strip(), val.strip()
        version_dict[key] = val
    return version_dict


# These things are picked up based on version.txt and README.md
VERSION_INFO = parse_version("version.txt")
DOWNLOAD_URL = VERSION_INFO["download_url"]
VERSION = VERSION_INFO["version"]

LONG_DESCRIPTION = read("README.md")
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

# Populate this with your information
DISTNAME = "package_name"
DESCRIPTION = "short_summary"
MAINTAINER = "you"
MAINTAINER_EMAIL = "your_email"
URL = "your_online_repo"
LICENSE = "short_string"  # For example "GNU-GPLv3"

# Include only strictly necessary packages here
# As a comma separated list of requirements
# Full requirements should be listed in requirements.txt
# Supports just "name", "name == version", or "name >= version".
INSTALL_REQUIRES = []

# List individual packages provided by your repository here
# Often this is just the single name, your_package
PACKAGES = [
    "your_package",
]

# A list of classifiers from https://pypi.org/classifiers/
# Some common classifiers are provided here
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Programming Language :: Python :: 3.8",
    "Operating System :: Unix",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS :: MacOS X",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
]

# You can also try except this to use distutils
from setuptools import setup

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
    )
