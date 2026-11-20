"""Deprecated command name; runs nested capability targets with shared settings."""
import argparse

from optimization_variable_n import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="capability_fixed_total")
    parser.add_argument("--restarts", type=int, default=1)
    args = parser.parse_args()
    run(args.run_id, criteria=("linf_area",), restarts=args.restarts)


if __name__ == "__main__":
    main()
