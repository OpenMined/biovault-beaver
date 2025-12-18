"""Security policy for deserializing beaver envelopes."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BeaverPolicy:
    """
    Deserialization policy with logging and security hooks.

    This policy logs all deserialization events and provides hooks
    to allow/block specific operations.
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        allow_functions: bool = True,
        allow_classes: Optional[set[str]] = None,
        block_modules: Optional[set[str]] = None,
    ):
        """
        Initialize the policy.

        Args:
            verbose: If True, print detailed logs to console
            allow_functions: Allow deserialization of functions/closures
            allow_classes: Set of allowed class names (None = allow all)
            block_modules: Set of blocked module names (e.g., {'subprocess', 'os'})
        """
        self.verbose = verbose
        self.allow_functions = allow_functions
        self.allow_classes = allow_classes
        self.block_modules = block_modules or {
            "subprocess",
            "os.system",
            "__builtin__.eval",
            "__builtin__.exec",
        }

    def _log(self, level: str, msg: str, **kwargs):
        """Log a message and optionally print if verbose."""
        log_fn = getattr(logger, level)
        log_fn(msg, extra=kwargs)
        if self.verbose:
            print(f"[BEAVER {level.upper()}] {msg}")
            if kwargs:
                for k, v in kwargs.items():
                    print(f"  {k}: {v}")

    def validate_class(self, cls: type, is_local: bool = False) -> Optional[type]:
        """
        Called before deserializing a class instance.

        Args:
            cls: The class about to be instantiated
            is_local: Whether this is a locally defined class

        Returns:
            The class to use (or None/raise to block)
        """
        module = cls.__module__
        qualname = cls.__qualname__
        full_name = f"{module}.{qualname}"

        self._log(
            "info",
            "Validating class",
            module=module,
            class_name=qualname,
            full_name=full_name,
            is_local=is_local,
        )

        # Check blocked modules
        for blocked in self.block_modules:
            if module.startswith(blocked):
                self._log("warning", f"BLOCKED: Module {module} is in blocklist")
                raise ValueError(f"Blocked module: {module}")

        # Check allowlist if specified
        if self.allow_classes is not None:
            allowed = full_name in self.allow_classes or qualname in self.allow_classes
            if not allowed:
                self._log("warning", f"BLOCKED: Class {full_name} not in allowlist")
                raise ValueError(f"Class not in allowlist: {full_name}")

        return cls

    # pyfory hook name
    def authorize_instantiation(self, cls: type, is_local: bool = False) -> Optional[type]:
        """Alias for validate_class expected by serializers."""
        return self.validate_class(cls, is_local=is_local)

    def validate_function(self, func: Callable, is_local: bool = False) -> Optional[Callable]:
        """
        Called before deserializing a function.

        Args:
            func: The function about to be loaded
            is_local: Whether this is a locally defined function

        Returns:
            The function to use (or None/raise to block)
        """
        module = getattr(func, "__module__", "unknown")
        name = getattr(func, "__name__", "unknown")
        qualname = getattr(func, "__qualname__", name)

        self._log(
            "info",
            "Validating function",
            module=module,
            name=name,
            qualname=qualname,
            is_local=is_local,
        )

        if not self.allow_functions:
            self._log("warning", "BLOCKED: Functions not allowed by policy")
            raise ValueError("Functions not allowed by policy")

        # Check blocked modules
        for blocked in self.block_modules:
            if module.startswith(blocked):
                self._log("warning", f"BLOCKED: Function from module {module} is blocked")
                raise ValueError(f"Blocked module: {module}")

        return func

    def validate_reduce(self, func: Callable, args: tuple) -> tuple:
        """
        Called before calling __reduce__ during deserialization.

        Args:
            func: The callable returned by __reduce__
            args: Arguments that will be passed to the callable

        Returns:
            (func, args) tuple to use (or raise to block)
        """
        func_name = getattr(func, "__name__", str(func))
        module = getattr(func, "__module__", "unknown")

        self._log(
            "info",
            "Validating __reduce__",
            func=func_name,
            module=module,
            args=str(args)[:100],
        )

        # Check blocked modules
        for blocked in self.block_modules:
            if module.startswith(blocked):
                self._log("warning", f"BLOCKED: __reduce__ from module {module}")
                raise ValueError(f"Blocked __reduce__ from module: {module}")

        return (func, args)

    def intercept_reduce_call(self, func: Callable, args: tuple) -> Optional[Any]:
        """
        Called right before executing the __reduce__ callable.

        Args:
            func: The callable from __reduce__
            args: Arguments to pass to the callable

        Returns:
            - None: Let normal flow continue (call func(*args))
            - Any object: Use this object instead (skip the call)
        """
        func_name = getattr(func, "__name__", str(func))
        module = getattr(func, "__module__", "unknown")

        self._log(
            "info",
            "Intercepting __reduce__ call",
            func=func_name,
            module=module,
            args=str(args)[:100],
        )

        # Return None to allow normal execution
        # Could return a different object here to intercept
        return None

    def inspect_reduced_object(self, obj: Any) -> Optional[Any]:
        """
        Called after __reduce__ has created the object.

        Args:
            obj: The object that was created via __reduce__

        Returns:
            - None: Use the object as-is
            - Any object: Replace with this object
        """
        obj_type = type(obj).__name__
        module = type(obj).__module__

        self._log(
            "info",
            "Inspecting reduced object",
            type=obj_type,
            module=module,
        )

        # Return None to accept the object
        # Could return a modified/different object here
        return None

    def validate_setstate(self, obj: Any, state: Any) -> Any:
        """
        Called before __setstate__ to sanitize/modify object state.

        Args:
            obj: The object being deserialized
            state: The state about to be applied

        Returns:
            Modified state (or original if no changes, or raise to block)
        """
        obj_type = type(obj).__name__
        module = type(obj).__module__

        self._log(
            "info",
            "Validating setstate",
            type=obj_type,
            module=module,
            state_type=type(state).__name__,
            state_keys=list(state.keys()) if isinstance(state, dict) else "N/A",
        )

        # Could filter out dangerous keys here
        # For now, just log and pass through
        return state

    # Alias expected by some serializers
    def intercept_setstate(self, obj: Any, state: Any) -> Any:
        return self.validate_setstate(obj, state)


# Default policy instances
PERMISSIVE_POLICY = BeaverPolicy(verbose=False, allow_functions=True)
VERBOSE_POLICY = BeaverPolicy(verbose=True, allow_functions=True)
STRICT_POLICY = BeaverPolicy(
    verbose=True,
    allow_functions=False,
    allow_classes={"builtins.dict", "builtins.list", "builtins.tuple"},
)

# Default policy: block functions; allow a vetted set of common data containers
DEFAULT_POLICY = BeaverPolicy(
    verbose=False,
    allow_functions=False,
    allow_classes={
        "builtins.dict",
        "builtins.list",
        "builtins.tuple",
        "builtins.set",
        "builtins.str",
        "builtins.int",
        "builtins.float",
        "builtins.bool",
        "builtins.bytes",
        "builtins.NoneType",
        # Beaver core types
        "Twin",
        "beaver.twin.Twin",
        "beaver.twin.CapturedFigure",
        "beaver.computation.ComputationRequest",
        "beaver.computation.ComputationResult",
        "beaver.computation.RemoteComputationPointer",
        "beaver.envelope.BeaverEnvelope",
        "beaver.live_var.LiveVar",
        "beaver.session.SessionRequest",
        "beaver.session.SessionDatasetInfo",
        "beaver.twin_func.TwinFunc",
        "beaver.twin_result.TwinComputationResult",
        "numpy.ndarray",
        "numpy.int64",
        "numpy.int32",
        "numpy.float64",
        "numpy.float32",
        "pandas.core.frame.DataFrame",
        "pandas.core.series.Series",
        "pandas.core.indexes.base.Index",
        "pandas.core.internals.managers.BlockManager",
    },
)

# Trusted policy: allow functions/classes (for explicitly trusted/local contexts)
TRUSTED_POLICY = BeaverPolicy(
    verbose=False,
    allow_functions=True,
    allow_classes=None,
)
