"""Optional Docker sandbox.

Nothing in this package may be imported eagerly when SANDBOX_MODE=disabled;
the manager imports it lazily only when the sandbox is enabled.
"""
