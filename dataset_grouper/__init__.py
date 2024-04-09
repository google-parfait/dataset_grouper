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
"""Dataset Grouper API."""

# When changing this, also move the unreleased changes in CHANGELOG.md to the
# associated version.
__version__ = '0.3.0'


from dataset_grouper import beam_transforms as beam
from dataset_grouper import serialization
from dataset_grouper.data_loaders import PartitionedDataset
from dataset_grouper.test_utils import prepare_test_tfrecord_dataset
from dataset_grouper.tfds_pipelines import tfds_group_counts
from dataset_grouper.tfds_pipelines import tfds_to_tfrecords
from dataset_grouper.types import Example
from dataset_grouper.types import GetKeyFn
