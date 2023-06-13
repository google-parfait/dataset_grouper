# Creating Group-Partitioned Datasets via Dataset Grouper

This directory contains examples of using Dataset Grouper to partition various
TFDS datasets. These are flag-based binaries that can be used to group and
partition various datasets.

## Basic Usage

We can run `group_by_feature.py` to partition a TFDS dataset by a feature. This
will create one group for each unique instance of the feature.

For example, we can run:

```
python -m group_by_feature.py \
  --tfds_name=mnist \
  --group_feature=label \
  --output_path=/tmp/dataset_grouper/mnist \
  --file_prefix="grouped_mnist.tfrecord" \
  --split="test"
```

This will partition `test` MNIST examples by their `label` feature, and write
the result to `/tmp/dataset_grouper/mnist`.

Note that `group_by_feature.py` downloads the dataset using
`dataset_builder.download_and_prepare()` and then created a group-partitioned
version of it. If you have already prepared the TFDS dataset, you can specify
the resulting directory via `--data_dir`.

## Large-Scale Language Modeling Datasets

Below we give suggested TFDS datasets and group features that will produce
useful larger scale grouped language modeling datasets. Note that
`group_by_feature.py` uses Beam's direct runner, which can be slow for larger
datasets. We highly recommend using
[more advanced runners](https://beam.apache.org/documentation/runners/capability-matrix/)
for these datasets.

### FedCCNews

* TFDS link: https://www.tensorflow.org/datasets/community_catalog/huggingface/cc_news
* TFDS name: `huggingface:cc_news`
* Group feature: `domain`
* Example command to create the dataset:

```
python -m group_by_feature.py \
  --tfds_name=huggingface:cc_news \
  --group_feature=domain \
  --output_path=/tmp/dataset_grouper/fedccnews \
  --file_prefix="fedccnews_train.tfrecord" \
  --split="train"
```

This downloads the dataset using `dataset_builder.download_and_prepare()`  and
then creates a group-partitioned version of it.

### FedBCO

* TFDS link: https://www.tensorflow.org/datasets/community_catalog/huggingface/bookcorpusopen
* TFDS name: `huggingface:bookcorpusopen`
* Group feature: `title`
* Example command to create the dataset:

```
python -m group_by_feature.py \
  --tfds_name=huggingface:bookcorpusopen \
  --group_feature=title \
  --output_path=/tmp/dataset_grouper/fedbco \
  --file_prefix="fedbco_train.tfrecord" \
  --split="train"
```

This downloads the dataset using `dataset_builder.download_and_prepare()`  and
then creates a group-partitioned version of it.

### FedWiki

* TFDS link: https://www.tensorflow.org/datasets/catalog/wiki40b
* TFDS name: `wikipedia/20220620.en`
* Group feature: `title`
* Command to create the dataset:

```
python -m group_by_feature.py \
  --tfds_name=wikipedia/20220620.en \
  --group_feature=title \
  --output_path=/tmp/dataset_grouper/fedwiki \
  --file_prefix="fedwiki_train.tfrecord" \
  --split="train"
```

This downloads the dataset using `dataset_builder.download_and_prepare()`  and
then creates a group-partitioned version of it.

### FedC4

* TFDS link: https://www.tensorflow.org/datasets/catalog/c4
* TFDS name: `c4`

Unlike the three datasets above, FedC4 is not split directly on a feature.
Instead, we extract the base domain in each example's URL (eg. `nytimes.com`)
and partition on this.

You can create FedC4 by following the instructions below.

1. **Download C4**: Due to the size of the C4 dataset, it must be downloaded and
prepared manually. Please follow the instructions on
[the TFDS C4 page](https://www.tensorflow.org/datasets/catalog/c4).
1. **Partition the dataset**: Run `group_c4.py`, making sure that you set
`--data_dir="${DATA_DIR}"` where `DATA_DIR` is the directory in which C4 was
prepared in the step above.

For example, you can run:

```
python -m group_c4.py \
  --data_dir="${DATA_DIR}" \
  --output_path=/tmp/dataset_grouper/fedc4 \
  --file_prefix="fedc4_train.tfrecord" \
  --split="train"
```



