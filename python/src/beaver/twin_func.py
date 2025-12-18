"""TwinFunc: Function binding for remote execution with local resolution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4

# Global registry: (func_id, owner) -> callable
_FUNC_REGISTRY: dict[tuple[str, str], Callable] = {}


def register_func(func: Callable, owner: str, name: Optional[str] = None) -> str:
    """
    Register a function and return its ID.

    Args:
        func: The callable to register
        owner: The owner's identity (email)
        name: Optional name for the function

    Returns:
        The function ID
    """
    func_name = name or getattr(func, "__name__", "anonymous")
    func_id = hashlib.sha256(f"{func_name}:{owner}:{uuid4().hex}".encode()).hexdigest()[:12]

    # Store metadata on the function
    func._beaver_func_id = func_id  # type: ignore
    func._beaver_owner = owner  # type: ignore
    func._beaver_name = func_name  # type: ignore

    # Add to registry
    _FUNC_REGISTRY[(func_id, owner)] = func

    return func_id


def get_func(func_id: str, owner: str) -> Optional[Callable]:
    """
    Get a function from the registry.

    Args:
        func_id: The function ID
        owner: The owner's identity

    Returns:
        The callable if found, None otherwise
    """
    return _FUNC_REGISTRY.get((func_id, owner))


def find_func_by_name(name: str, owner: str) -> Optional[Callable]:
    """
    Find a function by name and owner.

    Args:
        name: The function name
        owner: The owner's identity

    Returns:
        The callable if found, None otherwise
    """
    for (_fid, fowner), func in _FUNC_REGISTRY.items():
        if fowner == owner and getattr(func, "_beaver_name", func.__name__) == name:
            return func
    return None


@dataclass
class TwinFunc:
    """
    A function reference that can be passed across the network.

    When serialized, only the reference (ID, owner, name) is sent.
    When executed on the remote side, it resolves to the local function
    registered with the same name/ID.

    This enables callbacks that execute in the DATA OWNER's scope,
    with access to the DO's local variables (like progress Twins).

    Examples:
        # === DO Notebook ===
        progress = LiveVar({"epoch": 0}, name="progress")
        progress.enable_live()
        session.remote_vars["progress"] = progress

        # Define callback (executes in DO's scope!)
        @TwinFunc.bind
        def on_progress(epoch, total, acc):
            progress.value = {"epoch": epoch, "total": total, "acc": acc}

        session.remote_vars["on_progress"] = on_progress

        # === DS Notebook ===
        on_progress = session.peer_remote_vars["on_progress"].load()

        @bv
        def train(data, on_progress=None):
            for epoch in range(5):
                # ... training ...
                if on_progress:
                    on_progress(epoch, 5, accuracy)  # Calls DO's function!
            return {"model": weights}

        result = train(data, on_progress=on_progress)
        result.request_private()

        # When DO runs request.run(), on_progress resolves to DO's local
        # function which updates DO's progress Twin in real-time!
    """

    func_id: str
    owner: str
    name: str
    _local_func: Optional[Callable] = field(default=None, repr=False)

    @classmethod
    def bind(cls, func: Callable = None, *, owner: str = None, name: str = None):
        """
        Decorator to bind a function as a TwinFunc.

        Can be used as @TwinFunc.bind or @TwinFunc.bind(owner="x@y.com")

        Args:
            func: The function to bind
            owner: Override owner (defaults to current user)
            name: Override name (defaults to function name)

        Returns:
            TwinFunc instance
        """

        def decorator(f: Callable) -> TwinFunc:
            nonlocal owner, name

            # Auto-detect owner if not provided
            if owner is None:
                from .remote_data import get_current_owner

                owner = get_current_owner()

            # Use function name if not provided
            if name is None:
                name = getattr(f, "__name__", "anonymous")

            # Register the function
            func_id = register_func(f, owner, name)

            # Create TwinFunc with local reference
            twin_func = cls(
                func_id=func_id,
                owner=owner,
                name=name,
                _local_func=f,
            )

            return twin_func

        # Handle @TwinFunc.bind vs @TwinFunc.bind()
        if func is not None:
            return decorator(func)
        return decorator

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call the function.

        If we have a local reference, use it directly.
        Otherwise, try to resolve from registry.
        """
        func = self._resolve()
        if func is None:
            raise RuntimeError(
                f"TwinFunc '{self.name}' (ID: {self.func_id}) not found. "
                f"This function must be registered locally by {self.owner}."
            )
        return func(*args, **kwargs)

    def _resolve(self) -> Optional[Callable]:
        """Resolve to the actual callable."""
        # Use local reference if available
        if self._local_func is not None:
            return self._local_func

        # Try registry by ID
        func = get_func(self.func_id, self.owner)
        if func:
            self._local_func = func
            return func

        # Try registry by name (fallback)
        func = find_func_by_name(self.name, self.owner)
        if func:
            self._local_func = func
            return func

        return None

    def to_ref(self) -> dict:
        """
        Convert to a serializable reference.

        This is what gets sent over the network.
        """
        return {
            "_beaver_func_ref": True,
            "func_id": self.func_id,
            "owner": self.owner,
            "name": self.name,
        }

    @classmethod
    def from_ref(cls, ref: dict) -> TwinFunc:
        """
        Create from a serialized reference.

        Args:
            ref: The reference dict with _beaver_func_ref, func_id, owner, name

        Returns:
            TwinFunc instance (without local func reference)
        """
        return cls(
            func_id=ref["func_id"],
            owner=ref["owner"],
            name=ref["name"],
            _local_func=None,  # Will be resolved when called
        )

    @classmethod
    def is_func_ref(cls, val: Any) -> bool:
        """Check if a value is a function reference dict."""
        return isinstance(val, dict) and val.get("_beaver_func_ref") is True

    def __repr__(self) -> str:
        status = "✓ bound" if self._local_func else "📍 reference"
        return f"TwinFunc({self.name}, owner={self.owner}, {status})"


def resolve_func_ref(val: Any, context=None) -> Any:
    """
    Resolve a function reference to a local callable.

    This is called during computation execution to convert
    serialized function references back to actual functions.

    Args:
        val: The value to check/resolve
        context: BeaverContext for owner lookup

    Returns:
        The resolved callable if it's a func ref, otherwise the original value
    """
    if not TwinFunc.is_func_ref(val):
        return val

    func_id = val["func_id"]
    owner = val.get("owner")
    name = val.get("name", "unknown")

    # Determine the current user (executor)
    current_user = None
    if context:
        current_user = getattr(context, "user", None)

    # Try registry lookup with current user (executor owns the callback)
    if current_user:
        func = get_func(func_id, current_user)
        if func:
            return func

        # Try by name with current user
        func = find_func_by_name(name, current_user)
        if func:
            return func

    # Try with original owner (fallback)
    if owner:
        func = get_func(func_id, owner)
        if func:
            return func

        func = find_func_by_name(name, owner)
        if func:
            return func

    # Try session remote_vars by name
    if context and hasattr(context, "_active_session"):
        session = context._active_session
        if session and hasattr(session, "remote_vars") and name in session.remote_vars.vars:
            var = session.remote_vars.vars[name]
            # Check if it's a TwinFunc stored as a RemoteVar
            if hasattr(var, "_twin_func") and callable(var._twin_func):
                return var._twin_func

    # Return a TwinFunc that will fail on call with a clear error
    return TwinFunc.from_ref(val)


def func_to_ref(val: Any) -> Any:
    """
    Convert a callable to a reference for serialization.

    Args:
        val: The value to check/convert

    Returns:
        Reference dict if it's a TwinFunc or registered callable,
        otherwise the original value
    """
    # Already a TwinFunc
    if isinstance(val, TwinFunc):
        return val.to_ref()

    # Registered callable with beaver metadata
    if callable(val) and hasattr(val, "_beaver_func_id"):
        return {
            "_beaver_func_ref": True,
            "func_id": val._beaver_func_id,
            "owner": getattr(val, "_beaver_owner", "unknown"),
            "name": getattr(val, "_beaver_name", val.__name__),
        }

    return val
