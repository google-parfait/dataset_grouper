# Dataset Grouper - Scalable Dataset Pipelines for Group-Structured Learning

![PyPI version](https://img.shields.io/pypi/v/dataset_grouper)

Dataset Grouper is a library for creating, writing, and iterating over datasets
with group-level structure. It is primarily intended for creating large-scale
datasets for federated learning research.

## Installation

We recommend installing via PyPI. Please check the [PyPI](https://pypi.org/project/dataset-grouper/) page for up-to-date version requirements, including python version requirements.

```
pip install --upgrade pip
pip install dataset-grouper
```

## Getting Started

Below is a simple starting example, that partitions MNIST across 10 clients by
label.

First we import some necessary packages, including [Apache Beam](https://beam.apache.org/), which will be used to run the Dataset Grouper's pipelines.

```python
import apache_beam as beam
import dataset_grouper as dsgp
import tensorflow_datasets as tfds
```

Next, we download and prepare the MNIST dataset.

```python
dataset_builder = tfds.builder('mnist')
dataset_builder.download_and_prepare(...)
```

We now write a function that assigns each MNIST example a client identifier
(generally a `bytes` object). In this case, we will partition examples according
to their label, but you can use much more interesting partition functions as
well.

```python
def label_partition(x):
  label = x['label'].numpy()
  return str(label).encode('utf-8')
```

Finally, we build a pipeline that will partition MNIST according to this
function, and run it using Beam's
[Direct Runner](https://beam.apache.org/documentation/runners/direct/).

```python
mnist_pipeline = dsgp.tfds_to_tfrecords(
    dataset_builder=dataset_builder,
    split='test',
    get_key_fn=label_partition,
    file_path_prefix=...
)
with beam.Pipeline() as root:
  mnist_pipeline(root)
```

This will save a version of MNIST that has been partitioned according to labels
to a [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format.
We can also load it to iterate over client datasets.

```python
partitioned_dataset = dsgp.PartitionedDataset(
  file_pattern=...,
  tfds_features='mnist')

for group_dataset in partitioned_dataset.build_group_stream():
  pass
```

Generally, `PartitionedDataset.build_group_stream()` is a `tf.data.Dataset` that
yields datasets, each of which is contains all the examples held by one group.
If you'd like to use these datasets with NumPy, you can simply do:

```python
group_dataset_numpy = group_dataset.as_numpy_iterator()
```

## What Else?

The example above is primarily for educational purposes. MNIST is a relatively
small dataset, and can generally fit entirely into memory. For more interesting
examples, check out the examples folder.

Dataset Grouper is intended more for large-scale datasets, especially those
datasets that do not fit into memory. For these datasets, we recommend using
more sophisticated [Beam runners](https://beam.apache.org/documentation/runners/capability-matrix/), in order to partition the data in a distributed fashion.

## Disclaimers

This is not an officially supported Google product.

This is a utility library that downloads and prepares public datasets. We do
not host or distribute these datasets, vouch for their quality or fairness, or
claim that you have license to use the dataset. It is your responsibility to
determine whether you have permission to use the dataset under the dataset's
license.

If you're interested in learning more about responsible AI practices, please
see Google AI's [Responsible AI Practices](https://ai.google/education/responsible-ai-practices).

Dataset Grouper is Apache 2.0 licensed. See the [`LICENSE`](LICENSE) file. File.
