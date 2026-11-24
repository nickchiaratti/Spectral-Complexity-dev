#!/usr/bin/env python
"""
Unified Test Runner for Spectral Complexity Public Test Suite.

Usage:
    python tests/runner.py --tier all
    python tests/runner.py --tier 1
    python tests/runner.py --tier 2
    python tests/runner.py --tier 3
    python tests/runner.py --tier 4
    python tests/runner.py --feature F1
    python tests/runner.py --feature F6 --verbose
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent

TIER_FILES = {
    1: TESTS_DIR / "test_e2e_tier1_features.py",
    2: TESTS_DIR / "test_e2e_tier2_boundaries.py",
    3: TESTS_DIR / "test_e2e_tier3_integration.py",
    4: TESTS_DIR / "test_e2e_tier4_realworld.py",
}


def build_pytest_args(tier=None, feature=None, verbose=False):
    """Constructs the argument list for pytest execution."""
    args = ["-p", "no:dash"]

    if verbose:
        args.append("-v")
    else:
        args.append("-q")

    if feature:
        # Target specific feature in Tier 1 (e.g., F1 -> test_f1_)
        tier1_file = str(TIER_FILES[1])
        feat_clean = feature.lower().replace("-", "_")
        if not feat_clean.endswith("_"):
            feat_pattern = f"{feat_clean}_"
        else:
            feat_pattern = feat_clean
        args.extend([tier1_file, "-k", feat_pattern])
        return args

    if tier is None or tier == "all":
        # Target all tiers
        for t_file in TIER_FILES.values():
            args.append(str(t_file))
    else:
        try:
            t_num = int(tier)
            if t_num in TIER_FILES:
                args.append(str(TIER_FILES[t_num]))
            else:
                raise ValueError(f"Invalid tier number: {tier}")
        except ValueError:
            raise ValueError(f"Invalid tier argument: {tier}. Must be 1, 2, 3, 4, or 'all'.")

    return args


def main():
    parser = argparse.ArgumentParser(description="Spectral Complexity Public Test Suite Runner")
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Test tier to execute (1: Features, 2: Boundaries, 3: Integration, 4: Real-World, all: Complete Suite)"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="Specific feature to test in Tier 1 (e.g., F1, F2, ..., F13)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose test output"
    )
    args = parser.parse_args()

    import pytest

    pytest_args = build_pytest_args(tier=args.tier, feature=args.feature, verbose=args.verbose)

    print("=" * 80)
    print(" SPECTRAL COMPLEXITY PUBLIC: TEST SUITE RUNNER")
    print("=" * 80)
    print(f" Target Tier:    {args.tier}")
    if args.feature:
        print(f" Target Feature: {args.feature}")
    print(f" Invocation:     pytest {' '.join(pytest_args)}")
    print("=" * 80)

    start_time = time.perf_counter()
    exit_code = pytest.main(pytest_args)
    duration = time.perf_counter() - start_time

    print("\n" + "=" * 80)
    print(f" EXECUTION SUMMARY: Completed in {duration:.2f} seconds")
    print(f" Exit Code: {exit_code} ({'SUCCESS' if exit_code in (0, 5) else 'FAILURE'})")
    print("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
