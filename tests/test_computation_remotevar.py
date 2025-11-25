import pickle
import sys
import tempfile
import types
from pathlib import Path

# Stub pyfory so tests can run without the dependency installed.
if "pyfory" not in sys.modules:
    class _StubFory:
        def __init__(self, *args, **kwargs):
            pass

        def register_type(self, *_args, **_kwargs):
            return None

        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    sys.modules["pyfory"] = types.SimpleNamespace(Fory=_StubFory)

sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "src"))

from beaver.computation import RemoteComputationPointer, StagingArea, execute_remote_computation
from beaver.remote_vars import RemoteVarRegistry
from beaver.twin import Twin


def test_staging_area_remove():
    staging = StagingArea(context=None)
    comp = RemoteComputationPointer(result_name="demo")
    staging.add(comp)

    staging.remove(comp)

    assert comp not in staging.staged


def test_execute_remote_computation_resolves_twin_from_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = RemoteVarRegistry(owner="bob", registry_path=Path(tmpdir) / "remote_vars.json")
        twin = Twin(private=[1, 2, 3], public=[0], owner="bob", name="dataset")
        registry.add(twin, name="dataset")

        # Ensure the registry holds a reference to the owner's Twin
        remote_var = registry.vars["dataset"]
        assert hasattr(remote_var, "_twin_reference")

        class Context:
            user = "bob"
            remote_vars = registry

        ref = {
            "_beaver_remote_var": True,
            "owner": "bob",
            "var_id": remote_var.var_id,
            "name": "dataset",
        }

        result = execute_remote_computation(
            func=lambda t: t.value,
            args=(ref,),
            kwargs={},
            result_var_name="result",
            result_var_id="result-id",
            context=Context(),
        )

        assert result["result"] == [1, 2, 3]
