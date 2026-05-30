from services.validator_worker.run import _validate_record
import pytest


def test_validate_record_ok():
    dim = 4
    import os

    os.environ["VECTOR_DIM"] = str(dim)
    obj = {"id": "a", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"k": 1}}
    out = _validate_record(obj)
    assert out["id"] == "a"
    assert len(out["values"]) == dim


def test_validate_record_bad_dim():
    import os

    os.environ["VECTOR_DIM"] = "3"
    with pytest.raises(ValueError):
        _validate_record({"id": "a", "values": [0.1, 0.2], "metadata": {}})


def test_validate_record_empty_metadata_is_left_empty():
    """An empty `{}` metadata is stored verbatim — no `__rb_empty__` sentinel.

    The Parquet landing writer now JSON-encodes metadata into a string column,
    so the validator no longer injects a sentinel field to make the (formerly
    struct) schema writable.
    """
    import os

    os.environ["VECTOR_DIM"] = "4"
    out = _validate_record({"id": "a", "values": [0.1, 0.2, 0.3, 0.4], "metadata": {}})
    assert out["metadata"] == {}
    assert "__rb_empty__" not in out["metadata"]

    # An omitted metadata key also defaults to a plain empty dict.
    out2 = _validate_record({"id": "b", "values": [0.1, 0.2, 0.3, 0.4]})
    assert out2["metadata"] == {}

