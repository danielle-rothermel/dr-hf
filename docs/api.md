# API Reference

Auto-generated API documentation is available via pdoc.

## Generating Docs

### Interactive Server

```bash
uv run pdoc dr_hf
```

Opens a local server at http://localhost:8080 with interactive documentation.

### Static HTML

```bash
uv run pdoc dr_hf -o docs/api_html
```

Generates static HTML documentation in `docs/api_html/`.

## Module Index

- `dr_hf.pins` - dataset and model pins, commit SHA validation
- `dr_hf.datasets` - pinned dataset reader
- `dr_hf.subsets` - named subsets, derivations, registry
- `dr_hf.hashing` - `content_hash`
- `dr_hf.publishing` - model publishing with provenance
- `dr_hf.branches` - checkpoint branch discovery and parsing
- `dr_hf.io` - atomic dataset file commits
- `dr_hf.location` - `HFLocation` Pydantic model
- `dr_hf.models` - branch metadata and dataset commit models

## Public API

All public functions and models are exported from the top-level `dr_hf`
package; see `dr_hf.__all__` and the [README](../README.md).
