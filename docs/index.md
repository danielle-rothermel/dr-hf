# dr_hf Documentation

Hugging Face Hub identity layer: which dataset rows and which model weights,
fixed at commit SHAs. Loading models and running inference live in
dr-providers.

## Modules

### Datasets
- [pins](pins.md) - dataset and model pins resolved to commit SHAs
- [datasets](datasets.md) - all-split reader with origin split and native id
- [subsets](subsets.md) - named subsets, derivations, and the subset registry

### Models
- [branches](branches.md) - checkpoint branch discovery and `stepN-seed-*` parsing
- [publishing](publishing.md) - model upload with card, provenance, and tags

### Hub writes and locations
- [io](io.md) - atomic multi-file dataset commits
- [location](location.md) - `HFLocation` dataset repository references
- [models](models.md) - branch metadata and dataset commit models

## API Reference

See [API Reference](api.md) for auto-generated documentation.
