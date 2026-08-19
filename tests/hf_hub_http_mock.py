from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock, patch

from huggingface_hub import get_session
from huggingface_hub.utils.sha import git_hash

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ID = "test-org/test-dataset"
PARENT_OID = "91c54ad1727ee830252e457677f467be0bfd8a57"
NEW_COMMIT_OID = "c3d4e5f6789012345678901234567890ab12cd"


def revision_info(*, sha: str) -> dict[str, str]:
    return {"id": REPO_ID, "sha": sha}


def preupload_response(*, path: str, remote_oid: str | None) -> dict:
    return {
        "files": [
            {
                "path": path,
                "uploadMode": "regular",
                "shouldIgnore": False,
                "oid": remote_oid,
            }
        ]
    }


def commit_response(*, oid: str) -> dict[str, str]:
    return {
        "commitUrl": (
            f"https://huggingface.co/datasets/{REPO_ID}/commit/{oid}"
        ),
        "commitOid": oid,
    }


@contextmanager
def stub_hf_hub_http(
    *,
    repo_path: str,
    file_content: bytes,
    upload_case: Literal["changed", "unchanged"],
    parent_oid: str = PARENT_OID,
    new_commit_oid: str = NEW_COMMIT_OID,
) -> Iterator[list[tuple[str, str]]]:
    if upload_case == "unchanged":
        remote_oid: str | None = git_hash(file_content)
    else:
        remote_oid = None

    calls: list[tuple[str, str]] = []

    def fake_request(method: str, url: str, **kwargs: object) -> MagicMock:
        calls.append((method.upper(), url))
        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.text = ""
        if "/revision/" in url and method.upper() == "GET":
            response.json.return_value = revision_info(sha=parent_oid)
        elif "/preupload/" in url:
            response.json.return_value = preupload_response(
                path=repo_path,
                remote_oid=remote_oid,
            )
        elif "/commit/" in url:
            if upload_case == "unchanged":
                msg = "commit endpoint must not be called for unchanged upload"
                raise AssertionError(msg)
            response.json.return_value = commit_response(oid=new_commit_oid)
        else:
            response.json.return_value = revision_info(sha=parent_oid)
        return response

    with patch.object(get_session(), "request", side_effect=fake_request):
        yield calls
