# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -*- encoding: utf-8 -*-
import os

from setuptools import find_namespace_packages
from setuptools import find_packages
from setuptools import setup

this_directory = os.path.dirname(__file__)
req_path = os.path.join(this_directory, "requirements.txt")
req_dev_path = os.path.join(this_directory, "requirements_dev.txt")

install_requires = []
if os.path.exists(req_path):
    with open(req_path) as fp:
        install_requires = [line.strip() for line in fp]

install_dev_requires = []
if os.path.exists(req_dev_path):
    with open(req_dev_path) as fp:
        install_dev_requires = [line.strip() for line in fp]

readme_path = os.path.join(this_directory, "README.md")
readme_contents = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fp:
        readme_contents = fp.read().strip()

with open(os.path.join(this_directory, "orthogonium/VERSION")) as f:
    version = f.read().strip()

setup(
    # Name of the package:
    name="orthogonium",
    # Version of the package:
    version=version,
    # Find the package automatically (include everything):
    packages=find_packages(include=["orthogonium", "orthogonium.*"]),
    package_data={
        "orthogonium": ["VERSION"],  # Add the VERSION file
    },
    # Author information:
    author="Thibaut Boissin",
    author_email="thibaut.boissin@gmail.com",
    # Description of the package:
    description="",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    # Plugins entry point
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    licence="MIT",
    install_requires=install_requires,
    extras_require={
        "dev": install_dev_requires,
    },
)
