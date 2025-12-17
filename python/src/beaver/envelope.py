from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BeaverEnvelope:
    """Metadata + payload for a .beaver file."""

    version: int = 1
    envelope_id: str = field(default_factory=lambda: uuid4().hex)
    sender: str = "unknown"
    created_at: str = field(default_factory=_iso_now)
    name: Optional[str] = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    payload: bytes = b""
    reply_to: Optional[str] = None

    def filename(self, *, suffix: str = ".beaver") -> str:
        """Generate a sortable filename using uuid7 if available, else uuid4."""
        try:
            import uuid

            if hasattr(uuid, "uuid7"):
                name = uuid.uuid7().hex  # type: ignore[attr-defined]
            else:
                name = UUID(self.envelope_id).hex if self.envelope_id else uuid4().hex
        except Exception:
            name = self.envelope_id
        return f"{name}{suffix}"

    def load(
        self,
        *,
        inject: bool = True,
        overwrite: bool = True,
        globals_ns: Optional[dict] = None,
        strict: bool = False,
        policy=None,
        live: bool = True,
        context=None,
        auto_accept: bool = False,
        backend=None,
        trust_loader: bool | None = None,
    ) -> Any:
        """
        Load the envelope payload and inject into caller's globals.

        Args:
            inject: Whether to inject into globals
            overwrite: If False, prompt before overwriting existing variables
            globals_ns: Target namespace (auto-detected if None)
            strict: Strict deserialization mode
            policy: Deserialization policy
            live: If True and object is a live-enabled Twin, auto-subscribe for updates (default: True)
            context: BeaverContext for live subscription (auto-detected if None)
            auto_accept: If True, automatically accept trusted loaders without prompting
            backend: Optional SyftBoxBackend for reading encrypted artifact files
            trust_loader: If True, run loader in trusted mode. If None, prompt on failure.
        """
        import inspect

        from .runtime import _check_overwrite, _inject, unpack

        # Check for missing imports before unpacking (to avoid ModuleNotFoundError during deserialization)
        if self.manifest.get("required_versions"):
            self._check_and_prompt_missing_imports()

        # Auto-detect backend from context if not provided
        if (
            backend is None
            and context is not None
            and hasattr(context, "_backend")
            and context._backend
        ):
            backend = context._backend

        # Also try to get backend from session's context (for inbox-loaded envelopes)
        if backend is None and hasattr(self, "_session") and self._session is not None:
            session = self._session
            if hasattr(session, "_context") and session._context is not None:
                ctx = session._context
                if hasattr(ctx, "_backend") and ctx._backend:
                    backend = ctx._backend

        obj = unpack(
            self,
            strict=strict,
            policy=policy,
            auto_accept=auto_accept,
            backend=backend,
            trust_loader=trust_loader,
        )

        # Attach session reference if envelope was loaded from a session context
        if hasattr(self, "_session") and self._session is not None:
            obj._session = self._session

        # Auto-subscribe to live updates if this is a live-enabled Twin
        if live:
            from .twin import Twin

            if isinstance(obj, Twin) and hasattr(obj, "_live_enabled") and obj._live_enabled:
                # Auto-detect context if not provided
                if context is None:
                    # Try to find BeaverContext in caller's scope
                    frame = inspect.currentframe()
                    while frame and frame.f_back:
                        frame = frame.f_back
                        for scope in [frame.f_locals, frame.f_globals]:
                            for _var_name, var_obj in scope.items():
                                if (
                                    hasattr(var_obj, "remote_vars")
                                    and hasattr(var_obj, "user")
                                    and hasattr(var_obj, "inbox_path")
                                ):
                                    context = var_obj
                                    break
                            if context:
                                break
                        if context:
                            break

                if context:
                    # Start watching for live updates in background
                    print(f"📡 Auto-subscribed to live updates for Twin '{obj.name or 'unnamed'}'")
                    print(
                        "💡 Use .watch_live() to get updates, or reload with live=False to disable"
                    )
                    # Note: We don't actually start watching here, just inform the user
                    # They need to call watch_live() to get the generator

        if inject:
            if globals_ns is None:
                # Try to get caller's globals
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    globals_ns = frame.f_back.f_globals

                # Fallback for Jupyter notebooks
                if globals_ns is None or "__IPYTHON__" in globals_ns:
                    try:
                        import __main__

                        globals_ns = __main__.__dict__
                    except ImportError:
                        pass

            if globals_ns is not None:
                # Check for overwrites if needed
                if not overwrite:
                    should_proceed = _check_overwrite(
                        obj, globals_ns=globals_ns, name_hint=self.name
                    )
                    if not should_proceed:
                        print("⚠️  Load cancelled - no variables were overwritten")
                        return obj

                injected_names = _inject(obj, globals_ns=globals_ns, name_hint=self.name)
                if injected_names:
                    names_str = "', '".join(injected_names)
                    print(f"✓ Loaded '{names_str}' into globals")

        return obj

    def _check_and_prompt_missing_imports(self) -> None:
        """
        Check for missing imports and prompt user to install before unpacking.

        Raises ImportError if user declines to install missing packages.
        """
        required_versions = self.manifest.get("required_versions", {})
        if not required_versions:
            return

        # Check which modules are missing
        missing = []
        for module_name, version in required_versions.items():
            try:
                __import__(module_name)
            except ImportError:
                # Map common module names to pip package names
                package_map = {
                    "sklearn": "scikit-learn",
                    "cv2": "opencv-python",
                    "PIL": "pillow",
                    "skimage": "scikit-image",
                }
                pkg_name = package_map.get(module_name, module_name)
                missing.append((pkg_name, version))

        if not missing:
            return

        # Get function name from manifest if available
        func_name = self.manifest.get("func_name", self.name or "computation")

        # Prompt user to install
        from .computation import _is_uv_venv, _prompt_install_function_deps

        if not _prompt_install_function_deps(missing, func_name):
            # User declined - raise helpful error
            import shutil

            use_uv = _is_uv_venv() and shutil.which("uv")
            pip_cmd = "uv pip install" if use_uv else "pip install"
            specs = [f"{p}=={v}" if v else p for p, v in missing]
            raise ImportError(
                f"\n\n❌ Cannot load '{func_name}' - missing dependencies\n\n"
                f"   Required packages not installed:\n"
                + "\n".join(f"   • {s}" for s in specs)
                + f"\n\n   To fix, run:\n"
                f"   {pip_cmd} {' '.join(specs)}\n"
            )

    def __str__(self) -> str:
        """Human-readable representation of the envelope."""
        name = self.name or "(unnamed)"
        obj_type = self.manifest.get("type", "unknown")
        module = self.manifest.get("module")
        size = self.manifest.get("size_bytes", len(self.payload))
        envelope_type = self.manifest.get("envelope_type", "unknown")

        type_str = f"{obj_type}"
        if module and module != obj_type:
            type_str = f"{obj_type} ({module})"

        # Header with envelope type badge
        lines = [
            f"BeaverEnvelope [{envelope_type.upper()}]: {name}",
            f"  From: {self.sender}",
            f"  Type: {type_str}",
            f"  Size: {size} bytes",
            f"  Created: {self.created_at[:19].replace('T', ' ')} UTC",
            f"  ID: {self.envelope_id[:8]}...",
        ]

        if self.reply_to:
            lines.append(f"  Reply to: {self.reply_to[:8]}...")

        # Data-specific info
        if envelope_type == "data":
            shape = self.manifest.get("shape")
            dtype = self.manifest.get("dtype")
            columns = self.manifest.get("columns")
            preview = self.manifest.get("preview")

            if shape:
                lines.append(f"  Shape: {shape}")
            if dtype:
                lines.append(f"  Data type: {dtype}")
            if columns:
                lines.append(f"  Columns: {', '.join(columns)}")

            # Show preview
            if preview:
                lines.append("")
                lines.append("Preview:")
                lines.append(f"  {preview}")

        # Code-specific info
        elif envelope_type == "code":
            # Check if this is a ComputationRequest
            obj_type = self.manifest.get("type", "")
            if obj_type == "ComputationRequest":
                # Try to load and inspect the computation
                try:
                    from .runtime import unpack

                    comp_req = unpack(self, strict=False)
                    from .computation import _describe_bound_data

                    # Get context if we can (walk up frame stack)
                    context = None
                    import inspect

                    current = inspect.currentframe()
                    while current and current.f_back and context is None:
                        current = current.f_back
                        for scope in [current.f_locals, current.f_globals]:
                            for _var_name, var_obj in scope.items():
                                if (
                                    hasattr(var_obj, "remote_vars")
                                    and hasattr(var_obj, "user")
                                    and hasattr(var_obj, "inbox_path")
                                ):
                                    context = var_obj
                                    break
                            if context:
                                break

                    # Show arguments with detailed type info (similar to ComputationRequest.__repr__)
                    if comp_req.args:
                        lines.append("")
                        lines.append(f"Arguments ({len(comp_req.args)}):")
                        for i, arg in enumerate(comp_req.args):
                            # Check if this is a RemoteVar/Twin reference
                            if isinstance(arg, dict) and arg.get("_beaver_remote_var"):
                                twin_type = arg.get("var_type", "unknown")
                                is_twin = twin_type.startswith("Twin[")

                                if is_twin:
                                    # Try to find the actual Twin to check privacy status
                                    has_private = False
                                    has_public = False
                                    if context and arg["owner"] == context.user:
                                        for var in context.remote_vars.vars.values():
                                            if var.var_id == arg["var_id"]:
                                                from .twin import Twin

                                                if isinstance(var._stored_value, Twin):
                                                    has_private = var._stored_value.has_private
                                                    has_public = var._stored_value.has_public
                                                break

                                    # Show Twin with privacy indicator
                                    if has_private and has_public:
                                        privacy = "⚠️  REAL + MOCK"
                                    elif has_private:
                                        privacy = "🔒 PRIVATE"
                                    elif has_public:
                                        privacy = "🌍 PUBLIC"
                                    else:
                                        privacy = "⏳ PENDING"

                                    lines.append(
                                        f"  [{i}] {privacy} Twin: {arg['name']} "
                                        f"(type: {twin_type}, owner: {arg['owner']})"
                                    )
                                    if has_public:
                                        lines.append("      📊 Mock data available for testing")
                                    if has_private and context and arg["owner"] == context.user:
                                        lines.append(
                                            "      🔐 Real data available (you're the owner)"
                                        )
                                else:
                                    # Regular RemoteVar
                                    lines.append(
                                        f"  [{i}] RemoteVar: {arg['name']} "
                                        f"(type: {twin_type}, owner: {arg['owner']})"
                                    )
                            # Check if it's a Twin object directly (shouldn't happen but handle it)
                            elif hasattr(arg, "__class__") and arg.__class__.__name__ == "Twin":
                                from .twin import Twin

                                if isinstance(arg, Twin):
                                    from .runtime import _strip_ansi_codes

                                    # Twin should have been serialized, but handle it anyway
                                    if arg.has_private and arg.has_public:
                                        privacy = "⚠️  REAL + MOCK"
                                    elif arg.has_private:
                                        privacy = "🔒 PRIVATE"
                                    elif arg.has_public:
                                        privacy = "🌍 PUBLIC"
                                    else:
                                        privacy = "⏳ PENDING"
                                    lines.append(
                                        f"  [{i}] {privacy} Twin: {arg.name} "
                                        f"(type: {arg.var_type}, owner: {arg.owner})"
                                    )
                            else:
                                # Static bound value
                                arg_type = type(arg).__name__
                                from .runtime import _strip_ansi_codes

                                arg_repr = _strip_ansi_codes(repr(arg))
                                if len(arg_repr) > 60:
                                    arg_repr = arg_repr[:57] + "..."
                                lines.append(f"  [{i}] {arg_type}: {arg_repr}")
                                lines.append("      📌 Static value (bound at call time)")

                    # Show kwargs if any
                    if comp_req.kwargs:
                        lines.append("")
                        lines.append(f"Keyword Arguments ({len(comp_req.kwargs)}):")
                        for k, v in comp_req.kwargs.items():
                            v_type = type(v).__name__
                            from .runtime import _strip_ansi_codes

                            v_repr = _strip_ansi_codes(repr(v))
                            if len(v_repr) > 60:
                                v_repr = v_repr[:57] + "..."
                            lines.append(f"  {k}= {v_type}: {v_repr}")
                            lines.append("      📌 Static value (bound at call time)")

                    # Show bound data summary (similar to ComputationRequest.__repr__)
                    bound_data_lines = _describe_bound_data(comp_req.args, comp_req.kwargs, context)
                    if bound_data_lines:
                        lines.append("")
                        lines.append("Bound Data:")
                        lines.extend(bound_data_lines)

                except Exception as e:
                    lines.append("")
                    lines.append("Arguments:")
                    lines.append(f"  (could not load: {e})")

            # Add signature info if available
            signature = self.manifest.get("signature")
            if signature:
                lines.append("")
                lines.append(f"Signature: {name}{signature}")

            # Add source code if available
            source = self.manifest.get("source")
            if source:
                lines.append("")
                lines.append("Source:")
                for line in source.rstrip().split("\n"):
                    lines.append(f"  {line}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Use string representation for repr as well."""
        return self.__str__()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        name = self.name or "(unnamed)"
        obj_type = self.manifest.get("type", "unknown")
        module = self.manifest.get("module")
        size = self.manifest.get("size_bytes", len(self.payload))
        envelope_type = self.manifest.get("envelope_type", "unknown")

        type_str = f"{obj_type}"
        if module and module != obj_type:
            type_str = f"{obj_type} <code>({module})</code>"

        # Color code by envelope type
        border_color = "#2196F3" if envelope_type == "data" else "#4CAF50"
        badge_color = "#2196F3" if envelope_type == "data" else "#4CAF50"

        html = [
            f"<div style='font-family: monospace; border-left: 3px solid {border_color}; padding-left: 10px;'>",
            f"<b>BeaverEnvelope</b> <span style='background: {badge_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px;'>{envelope_type.upper()}</span> {name}<br>",
            f"<b>From:</b> {self.sender}<br>",
            f"<b>Type:</b> {type_str}<br>",
            f"<b>Size:</b> {size} bytes<br>",
            f"<b>Created:</b> {self.created_at[:19].replace('T', ' ')} UTC<br>",
            f"<b>ID:</b> <code>{self.envelope_id[:8]}...</code>",
        ]

        if self.reply_to:
            html.append(f"<br><b>Reply to:</b> <code>{self.reply_to[:8]}...</code>")

        # Data-specific info
        if envelope_type == "data":
            shape = self.manifest.get("shape")
            dtype = self.manifest.get("dtype")
            columns = self.manifest.get("columns")
            preview = self.manifest.get("preview")

            if shape:
                html.append(f"<br><b>Shape:</b> {shape}")
            if dtype:
                html.append(f"<br><b>Data type:</b> {dtype}")
            if columns:
                html.append(f"<br><b>Columns:</b> {', '.join(columns)}")

            # Show preview
            if preview:
                import html as html_module

                escaped_preview = html_module.escape(preview)
                html.append("<br><br><b>Preview:</b>")
                html.append(
                    f"<pre style='background: rgba(128, 128, 128, 0.1); "
                    f"border: 1px solid rgba(128, 128, 128, 0.3); "
                    f"padding: 10px; margin-top: 5px; border-radius: 4px;'>{escaped_preview}</pre>"
                )

        # Code-specific info
        elif envelope_type == "code":
            # Add signature info if available
            signature = self.manifest.get("signature")
            if signature:
                html.append(f"<br><br><b>Signature:</b> <code>{name}{signature}</code>")

            # Add source code if available with syntax highlighting
            source = self.manifest.get("source")
            if source:
                try:
                    from pygments import highlight
                    from pygments.formatters import HtmlFormatter
                    from pygments.lexers import PythonLexer

                    formatter = HtmlFormatter(style="monokai", noclasses=True)
                    highlighted = highlight(source, PythonLexer(), formatter)
                    html.append("<br><br><b>Source:</b>")
                    html.append(
                        f"<div style='border: 1px solid rgba(128, 128, 128, 0.3); "
                        f"border-radius: 4px; overflow: hidden; margin-top: 5px;'>{highlighted}</div>"
                    )
                except ImportError:
                    # Fallback if pygments not available
                    import html as html_module

                    escaped_source = html_module.escape(source)
                    html.append("<br><br><b>Source:</b>")
                    html.append(
                        f"<pre style='background: rgba(128, 128, 128, 0.1); "
                        f"border: 1px solid rgba(128, 128, 128, 0.3); "
                        f"padding: 10px; margin-top: 5px; border-radius: 4px;'>{escaped_source}</pre>"
                    )

        html.append("</div>")
        return "".join(html)

    def source(self) -> Optional[str]:
        """Get the source code if available in manifest."""
        return self.manifest.get("source")

    def show_source(self) -> None:
        """Print the source code if available."""
        source = self.source()
        if source:
            print(source)
        else:
            print("No source code available in envelope")
