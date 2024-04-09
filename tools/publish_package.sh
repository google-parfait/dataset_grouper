#!/usr/bin/env bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Create a Python environment.
python3 -m venv "venv"
source "venv/bin/activate"
python --version
pip install --upgrade "pip"
pip --version

# Build the Python package.
pip install --upgrade build
# Use the default arguments to build both the source distribution (sdist) and
# the binary distribution (wheel). This builds the wheel from the sdist
# (instead of from the srcdir) and the files of the sdist are controlled by the
# `MANIFEST.in`.
python -m build

# Publish the Python package.
pip install --upgrade twine
twine check "dist/"*".whl"
twine upload "dist/"*".whl"

# Cleanup
deactivate
