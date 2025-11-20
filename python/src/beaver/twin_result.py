"""Twin computation result for privacy-preserving workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TwinComputationResult:
    """
    Result from running a computation on a Twin's public side.

    Allows requesting the computation to be run on the private side.
    """

    public_result: Any
    twin_id: str
    twin_name: Optional[str]
    owner: str
    func: Any
    args: tuple
    kwargs: dict
    context: Any = None

    def __repr__(self) -> str:
        """Show the public result with a hint about requesting private."""
        result_repr = repr(self.public_result)
        if len(result_repr) > 60:
            result_repr = result_repr[:57] + "..."

        twin_name = self.twin_name or f"Twin#{self.twin_id[:8]}"

        return (
            f"TwinComputationResult:\n"
            f"  🌍 Public result: {result_repr}\n"
            f"  From: {twin_name} (owner: {self.owner})\n"
            f"  💡 Call .request_private() to run on real data"
        )

    def __str__(self) -> str:
        """Use repr for string conversion."""
        return self.__repr__()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter."""
        result_repr = repr(self.public_result)
        if len(result_repr) > 60:
            result_repr = result_repr[:57] + "..."

        twin_name = self.twin_name or f"Twin#{self.twin_id[:8]}"

        return f"""
        <div style='font-family: monospace; border-left: 3px solid #4CAF50; padding-left: 10px;'>
            <b>TwinComputationResult</b><br>
            <table style='margin-top: 10px;'>
                <tr>
                    <td>🌍 Public result:</td>
                    <td><code>{result_repr}</code></td>
                </tr>
                <tr>
                    <td>From Twin:</td>
                    <td><b>{twin_name}</b></td>
                </tr>
                <tr>
                    <td>Owner:</td>
                    <td>{self.owner}</td>
                </tr>
                <tr>
                    <td colspan='2' style='padding-top: 10px;'>
                        💡 <a href='#'>Call .request_private()</a> to run on real data
                    </td>
                </tr>
            </table>
        </div>
        """

    @property
    def value(self):
        """Get the public result value."""
        return self.public_result

    def request_private(self):
        """
        Request to run the computation on the private side.

        Creates a computation request that the Twin owner can approve.
        """
        if self.context is None:
            print("⚠️  No context available - cannot send request")
            print("   (This happens when the computation wasn't run through a BeaverContext)")
            return

        # Create a computation with the Twin
        # The owner will see this and can approve it

        # Find the Twin in the owner's remote vars
        print(f"🔒 Requesting private computation from {self.owner}")
        print(f"   Twin: {self.twin_name or self.twin_id[:8]}...")
        print(f"   Function: {self.func.__name__}")
        print()
        print("💡 Next: Send a computation request to the owner")
        print("   They can run it on the real data and send back the result")
        print()

        # TODO: Actually create and send the computation request
        # For now, just show the intent
        print("   (Request flow implementation coming next!)")
