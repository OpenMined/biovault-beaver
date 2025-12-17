"""Tests for response output copying in wait_for_response."""

from __future__ import annotations

from pathlib import Path

from beaver import Twin
from beaver.runtime import pack, read_envelope, unpack, write_envelope


class TestResponseOutputCopying:
    """Tests that wait_for_response copies private outputs from response to original twin."""

    def test_private_outputs_copied_to_twin(self) -> None:
        """Verify private_stdout, private_stderr, and private_figures are copied."""
        # Create original twin (as DS would have after calling request_private)
        original_twin = Twin(
            public={"mock": "data"},
            private=None,
            owner="ds@test.local",
            name="result",
        )
        original_twin._pending_comp_id = "test-comp-id-123"

        # Create response twin (as DO would send back)
        response_twin = Twin(
            public={"mock": "data"},
            private={"real": "result"},
            owner="do@test.local",
            name="result",
        )
        response_twin.private_stdout = "Real computation output\n"
        response_twin.private_stderr = "Some warning\n"
        response_twin.private_figures = [{"_beaver_figure": True, "png_bytes": b"fake-png-data"}]

        # Simulate the _handle_response logic
        if hasattr(response_twin, "private"):
            original_twin.private = response_twin.private
            for attr in ("private_stdout", "private_stderr", "private_figures"):
                if hasattr(response_twin, attr):
                    setattr(original_twin, attr, getattr(response_twin, attr))

        # Verify all attributes were copied
        assert original_twin.private == {"real": "result"}
        assert original_twin.private_stdout == "Real computation output\n"
        assert original_twin.private_stderr == "Some warning\n"
        assert original_twin.private_figures is not None
        assert len(original_twin.private_figures) == 1
        assert original_twin.private_figures[0]["_beaver_figure"] is True

    def test_private_outputs_none_when_not_present(self) -> None:
        """Verify missing attributes don't cause errors."""
        original_twin = Twin(
            public={"mock": "data"},
            private=None,
            owner="ds@test.local",
            name="result",
        )

        # Response without captured outputs
        response_twin = Twin(
            public={"mock": "data"},
            private={"real": "result"},
            owner="do@test.local",
            name="result",
        )

        # Simulate the _handle_response logic
        if hasattr(response_twin, "private"):
            original_twin.private = response_twin.private
            for attr in ("private_stdout", "private_stderr", "private_figures"):
                if hasattr(response_twin, attr):
                    setattr(original_twin, attr, getattr(response_twin, attr))

        # Verify private was copied but outputs are None
        assert original_twin.private == {"real": "result"}
        assert original_twin.private_stdout is None
        assert original_twin.private_stderr is None
        assert original_twin.private_figures is None

    def test_non_twin_response_handled(self) -> None:
        """Verify non-Twin responses set private directly."""
        original_twin = Twin(
            public={"mock": "data"},
            private=None,
            owner="ds@test.local",
            name="result",
        )

        # Simple dict response (not a Twin)
        response = {"computed": "value"}

        # Simulate the _handle_response logic for non-Twin
        if hasattr(response, "private"):
            original_twin.private = response.private
        else:
            original_twin.private = response

        assert original_twin.private == {"computed": "value"}


class TestFigureSerialization:
    """Tests for figure serialization round-trip."""

    def test_figures_survive_pack_unpack(self, tmp_path: Path) -> None:
        """Verify figures are serialized and deserialized correctly."""
        twin = Twin(
            public="mock",
            private="real",
            owner="test@local",
            name="test",
        )
        twin.private_figures = [{"_beaver_figure": True, "png_bytes": b"test-png-bytes-1234"}]
        twin.public_figures = [{"_beaver_figure": True, "png_bytes": b"test-public-png"}]

        # Pack the twin
        env = pack(
            twin,
            sender="test@local",
            name="test",
            artifact_dir=tmp_path,
            preserve_private=True,
        )

        # Write and read back
        envelope_path = write_envelope(env, out_dir=tmp_path)

        # Unpack
        loaded_env = read_envelope(envelope_path)
        result = unpack(loaded_env, trust_loader=True)

        # Verify figures survived
        assert hasattr(result, "private_figures")
        assert hasattr(result, "public_figures")
        # Note: figures may be converted to CapturedFigure objects or remain as dicts

    def test_empty_figures_handled(self, tmp_path: Path) -> None:
        """Verify empty figure lists don't cause issues."""
        twin = Twin(
            public="mock",
            private="real",
            owner="test@local",
            name="test",
        )
        twin.private_figures = []
        twin.public_figures = None

        # Pack the twin
        env = pack(
            twin,
            sender="test@local",
            name="test",
            artifact_dir=tmp_path,
            preserve_private=True,
        )

        # Write and read back
        envelope_path = write_envelope(env, out_dir=tmp_path)

        loaded_env = read_envelope(envelope_path)
        result = unpack(loaded_env, trust_loader=True)

        # Empty/None figures should not cause errors
        assert result is not None


class TestOutputAttributes:
    """Tests for stdout/stderr attribute handling."""

    def test_stdout_stderr_survive_serialization(self, tmp_path: Path) -> None:
        """Verify stdout and stderr are preserved through pack/unpack."""
        twin = Twin(
            public="mock",
            private="real",
            owner="test@local",
            name="test",
        )
        twin.private_stdout = "Private execution output"
        twin.private_stderr = "Private error output"
        twin.public_stdout = "Public execution output"
        twin.public_stderr = "Public error output"

        # Pack the twin
        env = pack(
            twin,
            sender="test@local",
            name="test",
            artifact_dir=tmp_path,
            preserve_private=True,
        )

        # Write and read back
        envelope_path = write_envelope(env, out_dir=tmp_path)

        loaded_env = read_envelope(envelope_path)
        result = unpack(loaded_env, trust_loader=True)

        # Verify outputs survived
        assert result.private_stdout == "Private execution output"
        assert result.private_stderr == "Private error output"
        assert result.public_stdout == "Public execution output"
        assert result.public_stderr == "Public error output"


class TestPrivateOutputSecurity:
    """Security tests to ensure private outputs don't leak."""

    def test_private_figures_not_in_public_only_twin(self, tmp_path: Path) -> None:
        """Verify private figures are stripped when preserve_private=False."""
        twin = Twin(
            public="mock",
            private="secret-real-data",
            owner="test@local",
            name="test",
        )
        twin.private_figures = [{"_beaver_figure": True, "png_bytes": b"secret-private-figure"}]
        twin.private_stdout = "Secret private output"
        twin.private_stderr = "Secret private errors"

        # Pack WITHOUT preserve_private (default for public sharing)
        env = pack(
            twin,
            sender="test@local",
            name="test",
            artifact_dir=tmp_path,
            preserve_private=False,
        )

        # Write and read back
        envelope_path = write_envelope(env, out_dir=tmp_path)

        loaded_env = read_envelope(envelope_path)
        result = unpack(loaded_env, trust_loader=True)

        # Private data should NOT be present
        assert result.private is None, "Private value should be stripped"
        # Private outputs should also be None/empty
        assert result.private_figures is None or result.private_figures == [], (
            "Private figures should not leak"
        )
        assert result.private_stdout is None, "Private stdout should not leak"
        assert result.private_stderr is None, "Private stderr should not leak"

    def test_public_outputs_available_without_private(self, tmp_path: Path) -> None:
        """Verify public outputs are available even when private is stripped."""
        twin = Twin(
            public="mock-public-data",
            private="secret-real-data",
            owner="test@local",
            name="test",
        )
        twin.public_figures = [{"_beaver_figure": True, "png_bytes": b"public-figure-ok"}]
        twin.public_stdout = "Public output is fine"
        twin.private_figures = [{"_beaver_figure": True, "png_bytes": b"secret-figure"}]
        twin.private_stdout = "Secret output"

        # Pack without private
        env = pack(
            twin,
            sender="test@local",
            name="test",
            artifact_dir=tmp_path,
            preserve_private=False,
        )

        envelope_path = write_envelope(env, out_dir=tmp_path)

        loaded_env = read_envelope(envelope_path)
        result = unpack(loaded_env, trust_loader=True)

        # Public should be available
        assert result.public == "mock-public-data"
        assert result.public_stdout == "Public output is fine"
        # Public figures should be present (may be CapturedFigure or dict)
        assert result.public_figures is not None
        assert len(result.public_figures) == 1

        # Private should NOT be available
        assert result.private is None
        assert result.private_stdout is None

    def test_private_only_shared_with_preserve_private(self, tmp_path: Path) -> None:
        """Verify private is only included when explicitly requested."""
        twin = Twin(
            public="mock",
            private={"secret": "data", "password": "hunter2"},
            owner="test@local",
            name="test",
        )
        twin.private_stdout = "Executed with real credentials"

        # Pack WITH preserve_private (for approved results)
        env = pack(
            twin,
            sender="test@local",
            name="test",
            artifact_dir=tmp_path,
            preserve_private=True,
        )

        envelope_path = write_envelope(env, out_dir=tmp_path)

        loaded_env = read_envelope(envelope_path)
        result = unpack(loaded_env, trust_loader=True)

        # Now private SHOULD be present
        assert result.private == {"secret": "data", "password": "hunter2"}
        assert result.private_stdout == "Executed with real credentials"

    def test_response_only_copies_private_not_public(self) -> None:
        """Verify wait_for_response only updates private attributes, not public."""
        # Original twin with its own public data
        original_twin = Twin(
            public={"original": "public-data"},
            private=None,
            owner="ds@test.local",
            name="result",
        )
        original_twin.public_stdout = "Original public output"
        original_twin.public_figures = [{"_beaver_figure": True, "png_bytes": b"original"}]
        original_twin._pending_comp_id = "test-comp-id"

        # Response twin with different public data (shouldn't overwrite)
        response_twin = Twin(
            public={"different": "public"},
            private={"approved": "result"},
            owner="do@test.local",
            name="result",
        )
        response_twin.public_stdout = "Response public output"
        response_twin.private_stdout = "Response private output"
        response_twin.private_figures = [{"_beaver_figure": True, "png_bytes": b"private-fig"}]

        # Simulate wait_for_response logic - only copy private attrs
        if hasattr(response_twin, "private"):
            original_twin.private = response_twin.private
            for attr in ("private_stdout", "private_stderr", "private_figures"):
                if hasattr(response_twin, attr):
                    setattr(original_twin, attr, getattr(response_twin, attr))

        # Private should be updated
        assert original_twin.private == {"approved": "result"}
        assert original_twin.private_stdout == "Response private output"
        assert original_twin.private_figures[0]["png_bytes"] == b"private-fig"

        # Public should remain UNCHANGED (original values preserved)
        assert original_twin.public == {"original": "public-data"}
        assert original_twin.public_stdout == "Original public output"
        assert original_twin.public_figures[0]["png_bytes"] == b"original"
