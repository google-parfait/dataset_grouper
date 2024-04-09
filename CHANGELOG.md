# Changelog

<!--

Changelog follows the https://keepachangelog.com/ standard.

This allows:

* Auto-parsing release notes during automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Clickable headers in the rendered markdown.

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google-research/dataset_grouper/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

## [0.3.0] - 2024-04-09

* Updated README to include installation and usage instructions.
* Added standard open-source tooling, including package testing and requirements.
* Add `serialization` and `beam` modules to the API.
* Added utilities for creating test datasets.
* Added more examples, including a Dirichlet partitioning example.
* Removed `preprocess_fn` arg from `PartitionedDataset.build_group_stream`. Preprocessing should now be applied via `tf.data.Dataset.map`.

## [0.2.1] - 2023-06-12

* Fixed bug where beam pipelines would not work with datasets that were not globally prepared.

## [0.2.0] - 2023-06-12

* Improved parallelism in Beam pipelines to speed up dataset grouping.
* Updated pyproject.toml file for initial PyPI release.
* Added more detailed README file.

## [0.1.0] - 2023-05-31

* Initial release

[Unreleased]: https://github.com/google-research/dataset_grouper/compare/v0.3.0..HEAD
[0.2.0]: https://github.com/google-research/dataset_grouper/releases/tag/v0.2.1...v0.3.0
[0.2.1]: https://github.com/google-research/dataset_grouper/releases/tag/v0.2.0...v0.2.1
[0.2.0]: https://github.com/google-research/dataset_grouper/releases/tag/v0.1.0...v0.2.0
[0.1.0]: https://github.com/google-research/dataset_grouper/releases/tag/v0.1.0
