"""Dynamic-series modeling laboratory."""

__all__ = ["main"]

def main(argv: list[str] | None = None) -> None:
    from lab_1_model.main import main as run

    run(argv)