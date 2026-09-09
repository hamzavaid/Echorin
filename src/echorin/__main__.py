"""Application entry point."""

from __future__ import annotations


def main() -> int:
    """Start Echorin, falling back to a useful foundation message pre-GUI."""
    try:
        from echorin.gui.main_window import run_application
    except ImportError:
        print("Echorin is installed. GUI components are not yet available.")
        return 0
    return run_application()


if __name__ == "__main__":
    raise SystemExit(main())
