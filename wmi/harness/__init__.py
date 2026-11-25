"""Research harness: seeding, leakage guards, provenance, statistics.

Every experiment script goes through this package. Numbers that appear in a
paper must come from a JSON written by `provenance.save_result` -- never typed
by hand.
"""
from .leakage import LeakageError, assert_disjoint, observation_overlap, check_split
from .provenance import run_record, save_result, load_results
from .seeds import set_seed
from .stats import bootstrap_ci, wilson_ci

__all__ = [
    "LeakageError", "assert_disjoint", "observation_overlap", "check_split",
    "run_record", "save_result", "load_results", "set_seed", "bootstrap_ci", "wilson_ci",
]
