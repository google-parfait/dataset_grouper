# Copyright 2023 Google LLC
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
"""Common python types for type hints."""

from collections.abc import Callable, Sequence
import typing
from typing import Any, Union

# TODO(b/281123702): Remove this and import `collections.abc.Mapping`.
Mapping = typing.Mapping

Tensor = Any
NestedTensor = Union[
    Tensor,
    Sequence['NestedTensor'],
    Mapping[str, 'NestedTensor'],
]
Feature = Union[Tensor, Mapping[str, 'Feature']]
Example = Mapping[str, Feature]
GetKeyFn = Callable[[Example], bytes]
