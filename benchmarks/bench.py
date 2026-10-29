"""Run the benchmark suite from the command line."""

if __package__:
    from .runner import main
else:
    from runner import main


if __name__ == "__main__":
    raise SystemExit(main())
