#!/usr/bin/env python3
"""Smoke tests for the atomic-write helper backing tokens.bin/meta writers.

Run directly: `python3 tools/test_atomic_write.py` (no pytest dependency,
matches the rest of tools/ which is pytest-free). Verifies:
  * happy path: temp file is gone, final file has the bytes we wrote
  * mid-write exception: final path is untouched, no `.tmp` leak
  * existing final: replace is atomic (old contents survive on failure)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow `python3 tools/test_atomic_write.py` from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

from _tokens_bin import (  # noqa: E402
    EncodedDoc,
    _atomic_write,
    write_metadata,
    write_qids_sidecar,
    write_tokens_bin,
)


def test_happy_path_no_tmp_leak() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.bin"
        with _atomic_write(out, "wb") as f:
            f.write(b"hello world")
        assert out.exists(), "final path missing after successful write"
        assert out.read_bytes() == b"hello world"
        assert not (out.parent / "out.bin.tmp").exists(), "temp file leaked"


def test_exception_mid_write_leaves_no_partial() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.bin"
        try:
            with _atomic_write(out, "wb") as f:
                f.write(b"partial-")
                raise RuntimeError("simulated mid-write failure")
        except RuntimeError:
            pass
        assert not out.exists(), (
            "final path should not exist when writer body raised"
        )
        assert not (out.parent / "out.bin.tmp").exists(), (
            "tmp artifact must be cleaned up on exception"
        )


def test_existing_final_preserved_on_failure() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.bin"
        out.write_bytes(b"original-content")

        try:
            with _atomic_write(out, "wb") as f:
                f.write(b"new-content-")
                raise RuntimeError("simulated mid-write failure")
        except RuntimeError:
            pass

        assert out.read_bytes() == b"original-content", (
            "atomic-write must not clobber existing final on failure"
        )
        assert not (out.parent / "out.bin.tmp").exists()


def test_existing_final_replaced_on_success() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.bin"
        out.write_bytes(b"old")
        with _atomic_write(out, "wb") as f:
            f.write(b"new-content-XYZ")
        assert out.read_bytes() == b"new-content-XYZ"


def test_text_mode_with_encoding() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.json"
        with _atomic_write(out, "w", encoding="utf-8") as f:
            f.write('{"k": "café"}')
        assert out.read_text(encoding="utf-8") == '{"k": "café"}'


def test_write_tokens_bin_is_atomic() -> None:
    """Drive a real write through write_tokens_bin and confirm no .tmp leak."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "tokens.bin"
        encoded = [
            EncodedDoc(
                doc_id="d0",
                token_ids=[101, 1996, 102],
                vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ),
        ]
        stats = write_tokens_bin(out, encoded, dim=4)
        assert stats == {"n_docs": 1, "n_tokens": 3, "dim": 4}
        assert out.exists()
        assert not (out.parent / "tokens.bin.tmp").exists()
        # Header magic check sanity-pings the atomic rename produced a valid file.
        assert out.read_bytes()[:8] == b"TAC_TKN1"


def test_write_metadata_is_atomic() -> None:
    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "tokens.bin.meta.json"
        write_metadata(
            meta,
            mode="docs",
            model_name="test",
            dim=4,
            n_docs=1,
            n_tokens=3,
            doc_ids=["d0"],
            dropped=[],
            docs_jsonl=Path("docs.jsonl"),
            tool="tools/test_atomic_write.py",
            tool_version="0.0",
        )
        assert meta.exists()
        assert not (meta.parent / "tokens.bin.meta.json.tmp").exists()
        loaded = json.loads(meta.read_text(encoding="utf-8"))
        assert loaded["n_docs"] == 1


def test_write_qids_sidecar_is_atomic() -> None:
    with tempfile.TemporaryDirectory() as td:
        qids = Path(td) / "queries.bin.qids"
        write_qids_sidecar(qids, ["1", "2", "3"])
        assert qids.exists()
        assert not (qids.parent / "queries.bin.qids.tmp").exists()
        # 3 little-endian u32 = 12 bytes.
        assert qids.stat().st_size == 12


if __name__ == "__main__":
    tests = [
        test_happy_path_no_tmp_leak,
        test_exception_mid_write_leaves_no_partial,
        test_existing_final_preserved_on_failure,
        test_existing_final_replaced_on_success,
        test_text_mode_with_encoding,
        test_write_tokens_bin_is_atomic,
        test_write_metadata_is_atomic,
        test_write_qids_sidecar_is_atomic,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\n{len(tests)}/{len(tests)} tests passed")
