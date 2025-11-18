from __future__ import annotations

import base64
import collections.abc
import contextlib
import functools
import inspect
import json
import re
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
from uuid import uuid4

import pyfory

from .envelope import BeaverEnvelope


def _summarize(obj: Any) -> dict:
    # Get the underlying function if it's a partial
    func = obj
    if hasattr(obj, "func"):
        func = obj.func

    # Determine type - show function/class instead of partial if no args bound
    obj_type = type(obj).__name__
    if obj_type == "partial" and hasattr(obj, "func"):
        # Check if any args are actually bound
        has_bound_args = bool(obj.args or obj.keywords)
        if not has_bound_args:
            # No args bound, show as the underlying function type
            obj_type = type(func).__name__

    summary = {
        "type": obj_type,
        "module": getattr(func, "__module__", None) or getattr(obj, "__module__", None),
        "qualname": getattr(func, "__qualname__", None) or getattr(obj, "__qualname__", None),
    }

    # Determine envelope type: code or data
    is_code = inspect.isfunction(func) or inspect.isclass(func) or inspect.ismethod(func)
    summary["envelope_type"] = "code" if is_code else "data"

    if hasattr(obj, "__len__"):
        with contextlib.suppress(Exception):
            summary["len"] = len(obj)  # type: ignore[arg-type]

    # Add data-specific info
    if not is_code:
        with contextlib.suppress(Exception):
            # Capture preview/repr for data envelopes
            preview = repr(obj)
            # Limit preview size to 500 chars
            if len(preview) > 500:
                preview = preview[:497] + "..."
            summary["preview"] = preview

            # For numpy arrays
            if hasattr(obj, "shape"):
                summary["shape"] = str(obj.shape)
                summary["dtype"] = str(obj.dtype)
            # For pandas dataframes
            elif hasattr(obj, "columns"):
                summary["columns"] = list(obj.columns)
                summary["shape"] = str(obj.shape)

    # Try to capture source code and signature for code envelopes
    if is_code:
        with contextlib.suppress(Exception):
            # Capture signature for functions
            if inspect.isfunction(func):
                sig = inspect.signature(func)
                summary["signature"] = str(sig)

                # Check for return annotation
                if sig.return_annotation != inspect.Signature.empty:
                    return_type = sig.return_annotation
                    if hasattr(return_type, "__name__"):
                        summary["return_type"] = return_type.__name__
                    else:
                        summary["return_type"] = str(return_type)

            # Capture source code
            source = inspect.getsource(func)
            if source:
                # Strip decorators from source
                lines = source.splitlines()
                cleaned_lines = []
                skip_blank = False

                for line in lines:
                    stripped = line.lstrip()
                    # Skip decorator lines
                    if stripped.startswith("@"):
                        skip_blank = True  # Skip blank line after decorator too
                        continue
                    # Skip one blank line after decorator
                    if skip_blank and not stripped:
                        skip_blank = False
                        continue
                    skip_blank = False
                    cleaned_lines.append(line)

                # Remove leading blank lines
                while cleaned_lines and not cleaned_lines[0].strip():
                    cleaned_lines.pop(0)

                cleaned_source = "\n".join(cleaned_lines)
                summary["source"] = cleaned_source
                summary["source_lines"] = len(cleaned_lines)

    return summary


def pack(
    obj: Any,
    *,
    envelope_id: Optional[str] = None,
    sender: str = "unknown",
    name: Optional[str] = None,
    inputs: Optional[Iterable[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    requirements: Optional[Iterable[str]] = None,
    reply_to: Optional[str] = None,
    strict: bool = False,
    policy=None,
) -> BeaverEnvelope:
    """Serialize an object into a BeaverEnvelope (Python-native)."""
    fory = pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)
    payload = fory.dumps(obj)
    manifest = _summarize(obj)
    manifest["size_bytes"] = len(payload)
    manifest["language"] = "python"

    env_id = envelope_id or uuid4().hex

    return BeaverEnvelope(
        envelope_id=env_id,
        sender=sender,
        name=name,
        inputs=list(inputs or []),
        outputs=list(outputs or []),
        requirements=list(requirements or []),
        manifest=manifest,
        payload=payload,
        reply_to=reply_to,
    )


def _envelope_record(envelope: BeaverEnvelope) -> dict:
    return {
        "version": envelope.version,
        "envelope_id": envelope.envelope_id,
        "sender": envelope.sender,
        "created_at": envelope.created_at,
        "name": envelope.name,
        "inputs": envelope.inputs,
        "outputs": envelope.outputs,
        "requirements": envelope.requirements,
        "manifest": envelope.manifest,
        "reply_to": envelope.reply_to,
        "payload_b64": base64.b64encode(envelope.payload).decode("ascii"),
    }


def write_envelope(envelope: BeaverEnvelope, out_dir: Path | str = ".") -> Path:
    """Persist a .beaver file (JSON manifest + base64 payload)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / envelope.filename()
    path.write_text(json.dumps(_envelope_record(envelope), indent=2))
    return path


def read_envelope(path: Path | str) -> BeaverEnvelope:
    """Load a BeaverEnvelope from disk."""
    record = json.loads(Path(path).read_text())
    payload = base64.b64decode(record["payload_b64"])
    return BeaverEnvelope(
        version=record.get("version", 1),
        envelope_id=record.get("envelope_id"),
        sender=record.get("sender", "unknown"),
        created_at=record.get("created_at"),
        name=record.get("name"),
        inputs=record.get("inputs", []),
        outputs=record.get("outputs", []),
        requirements=record.get("requirements", []),
        manifest=record.get("manifest", {}),
        reply_to=record.get("reply_to"),
        payload=payload,
    )


def unpack(
    envelope: BeaverEnvelope,
    *,
    strict: bool = False,
    policy=None,
) -> Any:
    """Deserialize the payload in a BeaverEnvelope."""
    _install_builtin_aliases()
    fory = pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)
    return fory.loads(envelope.payload)


def _inject(obj: Any, *, globals_ns: dict, name_hint: Optional[str]) -> None:
    """Bind deserialized object into the provided namespace."""
    if isinstance(obj, dict):
        globals_ns.update(obj)
        return
    names = []
    if name_hint:
        names.append(name_hint)
    obj_name = getattr(obj, "__name__", None)
    if obj_name:
        names.append(obj_name)
    # Fallback to the type name if we have nothing else
    if not names:
        names.append(type(obj).__name__)
    for n in dict.fromkeys(names):  # preserve order, drop duplicates
        globals_ns[n] = obj


def _next_file(inbox: Path | str) -> Optional[Path]:
    inbox_path = Path(inbox)
    candidates = sorted(inbox_path.glob("*.beaver"))
    return candidates[0] if candidates else None


def find_by_id(inbox: Path | str, envelope_id: str) -> Optional[BeaverEnvelope]:
    """Locate and load a .beaver file by envelope_id."""
    inbox_path = Path(inbox)
    for path in sorted(inbox_path.glob("*.beaver")):
        with contextlib.suppress(Exception):
            record = json.loads(path.read_text())
            if record.get("envelope_id") == envelope_id:
                return read_envelope(path)
    return None


def load_by_id(
    inbox: Path | str,
    envelope_id: str,
    *,
    inject: bool = True,
    globals_ns: Optional[dict] = None,
    strict: bool = False,
    policy=None,
) -> Tuple[Optional[BeaverEnvelope], Optional[Any]]:
    """Load and inject an envelope payload by id into caller's globals."""
    env = find_by_id(inbox, envelope_id)
    if env is None:
        return None, None
    obj = unpack(env, strict=strict, policy=policy)
    if inject:
        if globals_ns is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                globals_ns = frame.f_back.f_globals
            else:
                globals_ns = globals()
        _inject(obj, globals_ns=globals_ns, name_hint=env.name)
    return env, obj


def list_inbox(inbox: Path | str) -> list[BeaverEnvelope]:
    """List and load all .beaver envelopes in an inbox."""
    inbox_path = Path(inbox)
    envelopes = []
    for p in sorted(inbox_path.glob("*.beaver")):
        with contextlib.suppress(Exception):
            envelopes.append(read_envelope(p))
    return envelopes


class InboxView:
    """Wrapper to pretty-print an inbox as a table in notebooks."""

    def __init__(self, inbox: Path | str, envelopes: list[BeaverEnvelope]) -> None:
        self.inbox = Path(inbox)
        self.envelopes = envelopes

    def __len__(self) -> int:
        return len(self.envelopes)

    def __getitem__(self, key):
        """Index by position (int), envelope_id (str), or name (str)."""
        if isinstance(key, int):
            return self.envelopes[key]
        elif isinstance(key, str):
            # Try matching by envelope_id first
            for env in self.envelopes:
                if env.envelope_id == key or env.envelope_id.startswith(key):
                    return env
            # Try matching by name
            for env in self.envelopes:
                if env.name == key:
                    return env
            raise KeyError(f"No envelope with id or name matching: {key}")
        elif isinstance(key, slice):
            return self.envelopes[key]
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def _rows(self):
        return [
            {
                "name": e.name or "(unnamed)",
                "id": e.envelope_id[:12] + "...",
                "sender": e.sender,
                "type": e.manifest.get("type"),
                "size_bytes": e.manifest.get("size_bytes"),
                "created_at": e.created_at[:19].replace("T", " "),
                "reply_to": e.reply_to[:12] + "..." if e.reply_to else "",
            }
            for e in self.envelopes
        ]

    def __repr__(self) -> str:
        rows = self._rows()
        if not rows:
            return f"InboxView({self.inbox}): empty"
        # simple text table
        headers = ["name", "id", "sender", "type", "size_bytes", "created_at", "reply_to"]
        lines = [" | ".join(headers)]
        for r in rows:
            lines.append(" | ".join(str(r.get(h, "")) for h in headers))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = self._rows()
        if not rows:
            return f"<b>InboxView({self.inbox}): empty</b>"
        headers = ["name", "id", "sender", "type", "size_bytes", "created_at", "reply_to"]
        html = ["<table>", "<thead><tr>"]
        html += [f"<th>{h}</th>" for h in headers]
        html += ["</tr></thead><tbody>"]
        for r in rows:
            html.append("<tr>")
            for h in headers:
                html.append(f"<td>{r.get(h, '')}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        return "".join(html)


def _filtered_namespace(ns: dict, include_private: bool) -> dict:
    filtered = {}
    for k, v in ns.items():
        if not include_private and k.startswith("_"):
            continue
        if k == "__builtins__":
            continue
        if isinstance(v, (types.ModuleType, types.FrameType)):
            continue
        if isinstance(v, re.Pattern):
            continue
        filtered[k] = v
    return filtered


def _serializable_subset(payload: dict, fory: pyfory.Fory) -> dict:
    """Return only items that can be serialized by the given fory instance."""
    serializable = {}
    for k, v in payload.items():
        try:
            fory.dumps(v)
            serializable[k] = v
        except Exception:
            continue
    return serializable


def save(
    path: Path | str,
    globals_ns: Optional[dict] = None,
    *,
    include_private: bool = False,
    strict: bool = False,
    policy=None,
) -> Path:
    """Capture a namespace to a specific .beaver file path."""
    ns = globals_ns if globals_ns is not None else globals()
    payload = _filtered_namespace(ns, include_private=include_private)
    fory = pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)
    payload = _serializable_subset(payload, fory)
    env = pack(
        payload,
        sender="snapshot",
        name="namespace",
        strict=strict,
        policy=policy,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_envelope_record(env), indent=2))
    return path


def load(
    path: Path | str,
    globals_ns: Optional[dict] = None,
    *,
    inject: bool = True,
    strict: bool = False,
    policy=None,
) -> Any:
    """Load a .beaver file and inject into caller's namespace."""
    env = read_envelope(path)
    _install_builtin_aliases()
    obj = unpack(env, strict=strict, policy=policy)
    if inject:
        if globals_ns is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                globals_ns = frame.f_back.f_globals
            else:
                globals_ns = globals()
        _inject(obj, globals_ns=globals_ns, name_hint=env.name)
    return obj


def snapshot(
    out_dir: Path | str = ".",
    globals_ns: Optional[dict] = None,
    *,
    include_private: bool = False,
    strict: bool = False,
    policy=None,
) -> Path:
    """Capture the current namespace to an auto-named .beaver file."""
    ns = globals_ns if globals_ns is not None else globals()
    payload = _filtered_namespace(ns, include_private=include_private)
    fory = pyfory.Fory(xlang=False, ref=True, strict=strict, policy=policy)
    payload = _serializable_subset(payload, fory)
    env = pack(
        payload,
        sender="snapshot",
        name="namespace",
        strict=strict,
        policy=policy,
    )
    return write_envelope(env, out_dir=out_dir)


def _install_builtin_aliases() -> None:
    """
    Install common collections.abc aliases into builtins for backward compatibility.

    Older dumps may reference global names like `Iterable` that are not in builtins.
    """
    import builtins
    import typing

    aliases = {
        "Iterable": collections.abc.Iterable,
        "Mapping": collections.abc.Mapping,
        "MutableMapping": collections.abc.MutableMapping,
        "Sequence": collections.abc.Sequence,
        "Set": collections.abc.Set,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "Any": typing.Any,
        "Tuple": typing.Tuple,
        "List": typing.List,
        "Dict": typing.Dict,
    }
    for name, obj in aliases.items():
        if not hasattr(builtins, name):
            setattr(builtins, name, obj)


def listen_once(
    *,
    inbox: Path | str,
    outbox: Optional[Path | str] = None,
    inject_globals: bool = False,
    autorun: bool = False,
    globals_ns: Optional[dict] = None,
    strict: bool = False,
    policy=None,
    delete_after: bool = False,
) -> Tuple[Optional[BeaverEnvelope], Optional[Any], Optional[Any]]:
    """
    Process the oldest .beaver file in inbox once.

    Returns (envelope, obj, result). If no file exists, all are None.
    """
    path = _next_file(inbox)
    if path is None:
        return None, None, None

    envelope = read_envelope(path)
    obj = unpack(envelope, strict=strict, policy=policy)

    target_globals = globals_ns if globals_ns is not None else globals()
    if inject_globals:
        _inject(obj, globals_ns=target_globals, name_hint=envelope.name)

    result = None
    if autorun and callable(obj):
        result = obj()

    if outbox is not None:
        reply_env = pack(
            result,
            sender="listener",
            name=f"{envelope.name}_result" if envelope.name else "result",
            reply_to=envelope.envelope_id,
            strict=strict,
            policy=policy,
        )
        write_envelope(reply_env, out_dir=outbox)

    if delete_after:
        path.unlink()

    return envelope, obj, result


@dataclass
class SendResult:
    path: Path
    envelope: BeaverEnvelope

    @property
    def envelope_id(self) -> str:
        return self.envelope.envelope_id


def export(
    *,
    sender: str = "unknown",
    out_dir: Path | str = "outbox",
    name: Optional[str] = None,
    inputs: Optional[Iterable[str]] = None,
    outputs: Optional[Iterable[str]] = None,
    requirements: Optional[Iterable[str]] = None,
    strict: bool = False,
    policy=None,
):
    """
    Decorator that adds .bind() and .send() to a callable.
    """

    def decorator(func):
        # Capture decorator's out_dir for use in BoundFunction
        default_out_dir = out_dir

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        class BoundFunction:
            """A function with bound arguments ready to send."""
            def __init__(self, bound_func, bound_args, bound_kwargs):
                self.bound_func = bound_func
                self.bound_args = bound_args
                self.bound_kwargs = bound_kwargs

            def send(self, *, user=None, out_dir=None):
                """Send the bound function to a destination."""
                partial = functools.partial(self.bound_func, *self.bound_args, **self.bound_kwargs)

                # Determine destination directory
                if out_dir is not None:
                    dest_dir = out_dir
                elif user is not None:
                    dest_dir = Path("shared") / user
                else:
                    dest_dir = default_out_dir

                env = pack(
                    partial,
                    sender=sender,
                    name=name or self.bound_func.__name__,
                    inputs=list(inputs or []),
                    outputs=list(outputs or []),
                    requirements=list(requirements or []),
                    strict=strict,
                    policy=policy,
                )
                path = write_envelope(env, out_dir=dest_dir)
                return SendResult(path=path, envelope=env)

        def bind(*args, **kwargs):
            """Bind arguments to the function for later sending."""
            return BoundFunction(func, args, kwargs)

        def send(*, user=None, out_dir=None):
            """Send the function (unbound) to a destination."""
            return bind().send(user=user, out_dir=out_dir)

        wrapper.bind = bind  # type: ignore[attr-defined]
        wrapper.send = send  # type: ignore[attr-defined]
        return wrapper

    return decorator


def wait_for_reply(
    *,
    inbox: Path | str,
    reply_to: str,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    strict: bool = False,
    policy=None,
    delete_after: bool = False,
) -> Tuple[Optional[BeaverEnvelope], Optional[Any]]:
    """
    Poll an inbox for a reply envelope whose reply_to matches.
    """
    inbox_path = Path(inbox)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path in sorted(inbox_path.glob("*.beaver")):
            try:
                record = json.loads(path.read_text())
            except Exception:
                continue
            if record.get("reply_to") not in {reply_to, Path(reply_to).name}:
                continue
            env = read_envelope(path)
            obj = unpack(env, strict=strict, policy=policy)
            if delete_after:
                path.unlink()
            return env, obj
        time.sleep(poll_interval)
    return None, None


class BeaverContext:
    """Connection helper holding identity and inbox/outbox defaults."""

    def __init__(
        self,
        *,
        inbox: Path | str,
        outbox: Optional[Path | str] = None,
        user: str = "unknown",
        biovault: Optional[str] = None,
        strict: bool = False,
        policy=None,
    ) -> None:
        self.inbox_path = Path(inbox)
        self.outbox = Path(outbox) if outbox is not None else self.inbox_path
        self.user = user
        self.biovault = biovault
        self.strict = strict
        self.policy = policy

    def __call__(self, func=None, **kwargs):
        """Allow @bv as a decorator; kwargs override context defaults."""
        def decorator(f):
            # capture base paths for user targeting inside the wrapper
            base_out = Path(kwargs.get("out_dir", self.outbox))
            base_parent = base_out.parent

            def wrapped_export(func):
                wrapped = export(
                    sender=kwargs.get("sender", self.user),
                    out_dir=base_out,
                    name=kwargs.get("name"),
                    inputs=kwargs.get("inputs"),
                    outputs=kwargs.get("outputs"),
                    requirements=kwargs.get("requirements"),
                    strict=kwargs.get("strict", self.strict),
                    policy=kwargs.get("policy", self.policy),
                )(func)

                original_send = wrapped.send

                def send_with_user(*, user=None, out_dir=None):
                    dest_dir = out_dir
                    if dest_dir is None and user:
                        dest_dir = base_parent / user
                    return original_send(user=user, out_dir=dest_dir)

                wrapped.send = send_with_user  # type: ignore[attr-defined]
                return wrapped

            return wrapped_export(f)

        if func is None:
            return decorator
        return decorator(func)

    def send(self, obj: Any, **kwargs) -> Path:
        """Pack and write an object using context defaults."""
        env = pack(
            obj,
            sender=kwargs.get("sender", self.user),
            name=kwargs.get("name"),
            inputs=kwargs.get("inputs"),
            outputs=kwargs.get("outputs"),
            requirements=kwargs.get("requirements"),
            reply_to=kwargs.get("reply_to"),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )
        out_dir = kwargs.get("out_dir", None)
        to_user = kwargs.get("user", None)
        if out_dir is None:
            if to_user:
                base = Path(self.outbox).parent
                out_dir = base / to_user
            else:
                out_dir = self.outbox
        path = write_envelope(env, out_dir=out_dir)
        return SendResult(path=path, envelope=env)

    def reply(self, obj: Any, *, to_id: str, **kwargs) -> Path:
        """Send a reply payload correlated to a prior envelope id."""
        return self.send(obj, reply_to=to_id, **kwargs)

    def listen_once(self, **kwargs):
        """Listen with context inbox/outbox defaults."""
        return listen_once(
            inbox=kwargs.get("inbox", self.inbox_path),
            outbox=kwargs.get("outbox", self.outbox),
            inject_globals=kwargs.get("inject_globals", False),
            autorun=kwargs.get("autorun", False),
            globals_ns=kwargs.get("globals_ns"),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
            delete_after=kwargs.get("delete_after", False),
        )

    def wait_for_reply(self, reply_to: str, **kwargs):
        """Wait for a reply matching reply_to in the context inbox."""
        return wait_for_reply(
            inbox=kwargs.get("inbox", self.inbox_path),
            reply_to=reply_to,
            timeout=kwargs.get("timeout", 30.0),
            poll_interval=kwargs.get("poll_interval", 0.5),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
            delete_after=kwargs.get("delete_after", False),
        )

    def load_by_id(self, envelope_id: str, **kwargs):
        """Load a payload by id using context inbox and inject into caller's globals."""
        inject = kwargs.get("inject", True)
        globals_ns = kwargs.get("globals_ns")

        # Auto-detect caller's globals if injecting and not provided
        if inject and globals_ns is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                globals_ns = frame.f_back.f_globals

        return load_by_id(
            inbox=kwargs.get("inbox", self.inbox_path),
            envelope_id=envelope_id,
            inject=inject,
            globals_ns=globals_ns,
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )

    def list_inbox(self, **kwargs):
        """List envelopes in the context inbox."""
        return list_inbox(kwargs.get("inbox", self.inbox_path))

    def inbox(self, sort_by: Optional[str] = None, reverse: bool = False, **kwargs):
        """Return an InboxView for the current inbox with optional sorting."""
        inbox_path = kwargs.get("inbox", self.inbox_path)
        envelopes = list_inbox(inbox_path)

        # Sort if requested
        if sort_by:
            def sort_key(env):
                if sort_by == "name":
                    return env.name or ""
                elif sort_by == "sender":
                    return env.sender
                elif sort_by == "created_at":
                    return env.created_at
                elif sort_by == "size":
                    return env.manifest.get("size_bytes", 0)
                elif sort_by == "type":
                    return env.manifest.get("type", "")
                return ""
            envelopes = sorted(envelopes, key=sort_key, reverse=reverse)

        return InboxView(inbox_path, envelopes)

    def save(self, path: Path | str, **kwargs) -> Path:
        """Save a namespace (default globals) using context defaults."""
        return save(
            path=path,
            globals_ns=kwargs.get("globals_ns"),
            include_private=kwargs.get("include_private", False),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )

    def load(self, path: Path | str, **kwargs) -> Any:
        """Load a .beaver file and inject into caller's namespace."""
        inject = kwargs.get("inject", True)
        globals_ns = kwargs.get("globals_ns")

        # Auto-detect caller's globals if injecting and not provided
        if inject and globals_ns is None:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                globals_ns = frame.f_back.f_globals

        return load(
            path=path,
            globals_ns=globals_ns,
            inject=inject,
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )

    def snapshot(self, **kwargs) -> Path:
        """Auto-name and write a namespace snapshot using context outbox."""
        return snapshot(
            out_dir=kwargs.get("out_dir", self.outbox),
            globals_ns=kwargs.get("globals_ns"),
            include_private=kwargs.get("include_private", False),
            strict=kwargs.get("strict", self.strict),
            policy=kwargs.get("policy", self.policy),
        )


def connect(
    folder: Path | str,
    *,
    user: str = "unknown",
    biovault: Optional[str] = None,
    inbox: Optional[Path | str] = None,
    outbox: Optional[Path | str] = None,
    strict: bool = False,
    policy=None,
) -> BeaverContext:
    """
    Create a BeaverContext with shared defaults.

    If inbox/outbox are omitted, both default to `folder`.
    """
    base = Path(folder)
    user_subdir = base / user if user else base
    return BeaverContext(
        inbox=inbox if inbox is not None else user_subdir,
        outbox=outbox if outbox is not None else user_subdir,
        user=user,
        biovault=biovault,
        strict=strict,
        policy=policy,
    )
