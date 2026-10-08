# publishing

Uploading a local model directory to a Hub model repository with its model
card and provenance in one commit.

## Types

```python
class ModelProvenance(BaseModel, frozen=True):
    source_repo_id: str
    source_revision: CommitSha            # commit of the original model
    conversion_tool: str                  # e.g. "datadec.models.convert 0.1.0"
    verification: Mapping[str, object]    # strict JSON, e.g. max_abs_logit_diff
    notes: str | None = None

class TagConflictError(RuntimeError): ...

MODEL_CARD_FILENAME = "README.md"
PROVENANCE_FILENAME = "provenance.json"
```

## publish_model

```python
def publish_model(
    local_dir, *, repo_id: str, branch: str, tags: Sequence[str] = (),
    card_markdown: str, provenance: ModelProvenance, private: bool,
    commit_message: str,
) -> ModelPin
```

1. Validates locally: `local_dir` is a non-empty directory without its own
   `README.md` or `provenance.json`, and the commit message and branch are
   non-blank.
2. `create_repo(repo_id, repo_type="model", private=private,
   exist_ok=True)`; `private` only applies when the repo is created.
3. `create_branch(exist_ok=True)` unless `branch` is `main`.
4. One `create_commit` on `branch` with every file under `local_dir`
   (top-level `.git` and `.cache` skipped), the card as `README.md`, and
   `provenance.json`. An unchanged upload creates no commit and the branch
   head is returned.
5. Each tag is created at that commit. A tag already at the commit is
   accepted; a tag at another commit raises `TagConflictError` (the commit
   has already landed).

Returns the `ModelPin` of the uploaded commit. Authentication uses the
standard `huggingface_hub` token resolution (`HF_TOKEN` or `hf auth login`).
