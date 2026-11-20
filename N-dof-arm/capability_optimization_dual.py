"""Deprecated alias for nested minimum-cost capability target comparisons."""
import argparse

from optimization_variable_n import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="capability_variable_n")
    parser.add_argument("--restarts", type=int, default=1)
    args = parser.parse_args()
    run(args.run_id, criteria=("linf_area",), restarts=args.restarts)


if __name__ == "__main__":
    main()
