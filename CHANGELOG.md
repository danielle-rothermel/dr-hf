# Changelog

## 0.1.2 - 2026-10-08

### Added

- `dr_hf.pins`: `DatasetPin` and `ModelPin` fixed at full commit SHAs
  (`CommitSha`), `ROW_INDEX` sentinel, `resolve_dataset_pin`, and
  `resolve_model_pin`.
- `dr_hf.datasets`: `SourceRow`, `read_rows` over every split at the pinned
  commit with native id validation (`NativeIdError`), and `split_sizes`.
- `dr_hf.subsets`: `SplitKey`, `DerivationKind`, `Derivation`,
  `PartitionShare`, `NamedSubset`, origin keys and subsets, seeded sample,
  seeded partition, exclusion, union, explicit ids, and the JSON-file
  `SubsetRegistry` with its one-key-one-content-hash rule.
- `dr_hf.hashing.content_hash`: the single hashing seam, to be replaced by
  dr-serialize.
- `dr_hf.publishing`: `ModelProvenance` and `publish_model`, which uploads a
  model directory with its card and `provenance.json` in one commit and
  creates tags without moving existing ones (`TagConflictError`).
- `hub` pytest marker for opt-in conformance tests against
  `allenai/ai2_arc` (ARC-Challenge) and `allenai/DataDecide-eval-instances`
  (sample-eval).
- `.defs` terms and contracts for pins, rows, subsets, the registry, model
  publication, and provenance.

### Changed

- Dependencies: `datasets>=5.1.0,<6`, `huggingface_hub>=1.31.0,<3`,
  `pydantic>=2.14.0,<3`; dev tools refreshed; `uv.lock` regenerated.
- `[tool.uv] required-version` is now `>=0.12.0`.
- All exports are eager; the lazy `__getattr__` loader is gone.

### Removed

- `download_dataset`, `load_or_download_dataset`, and `sanitize_repo_name`
  (replaced by `read_rows`).
- `dr_hf.weights`, `dr_hf.checkpoints`, `dr_hf.configs`, `dr_hf._torch`, and
  their models (`TensorStats`, `WeightsAnalysis`, `CheckpointAnalysis`,
  `ConfigAnalysis`, and the rest of the weight, optimizer, config, and
  checkpoint analysis models).
- `dr_hf.paths` (`get_data_dir`, `get_repo_dir`) and its import-time
  `.env` loading.
- `cached_download_tables_from_hf`, `get_tables_from_cache`,
  `read_local_parquet_paths`, and `query_hf_with_duckdb`.
- The `weights`, `duckdb`, and `all` extras and the `pandas` and
  `python-dotenv` dependencies.

## 0.1.1

- Atomic multi-file dataset commits (`commit_dataset_files_to_hf`) and
  release publishing infrastructure.
