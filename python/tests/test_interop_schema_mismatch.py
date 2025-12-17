from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from beaver.policy import TRUSTED_POLICY
from beaver.runtime import SecurityError, pack, unpack


def test_unpack_rejects_schema_mismatched_computation_request():
    """A ComputationRequest schema drift should be surfaced as a safe error."""
    import beaver.computation as comp_mod

    original = comp_mod.ComputationRequest

    @dataclass
    class ComputationRequest:  # noqa: N801 - must match receiver name
        comp_id: str
        result_id: str
        func: object
        args: tuple
        kwargs: dict
        sender: str
        result_name: str
        created_at: str = field(default_factory=lambda: "2025-01-01T00:00:00+00:00")
        schema_drift: str = "bad"

    # Ensure the serialized type resolves as beaver.computation.ComputationRequest on the receiver.
    ComputationRequest.__module__ = "beaver.computation"
    ComputationRequest.__qualname__ = "ComputationRequest"

    def f(_data: dict) -> dict:
        return {"ok": True}

    try:
        comp_mod.ComputationRequest = ComputationRequest  # type: ignore[assignment]
        req = comp_mod.ComputationRequest(  # type: ignore[call-arg]
            comp_id=uuid4().hex,
            result_id=uuid4().hex,
            func=f,
            args=(
                {"_beaver_remote_var": True, "name": "x", "owner": "me", "var_type": "Twin[dict]"},
            ),
            kwargs={},
            sender="sender@test.local",
            result_name="result",
        )
        env = pack(req, sender="sender@test.local", name="request_test", policy=TRUSTED_POLICY)
    finally:
        comp_mod.ComputationRequest = original  # type: ignore[assignment]

    with pytest.raises(SecurityError) as exc_info:
        _ = unpack(env, policy=TRUSTED_POLICY, auto_accept=True)

    msg = str(exc_info.value).lower()
    assert "hash" in msg and "not consistent" in msg and "computationrequest" in msg
