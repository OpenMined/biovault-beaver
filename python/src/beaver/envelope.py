from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
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
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)
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

    def load(self, *, inject: bool = True, globals_ns: Optional[dict] = None, strict: bool = False, policy=None) -> Any:
        """Load the envelope payload and inject into caller's globals."""
        from .runtime import unpack, _inject
        import inspect

        obj = unpack(self, strict=strict, policy=policy)

        if inject:
            if globals_ns is None:
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    globals_ns = frame.f_back.f_globals
            if globals_ns is not None:
                _inject(obj, globals_ns=globals_ns, name_hint=self.name)

        return obj

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
                    from pygments.lexers import PythonLexer
                    from pygments.formatters import HtmlFormatter

                    formatter = HtmlFormatter(style='monokai', noclasses=True)
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
