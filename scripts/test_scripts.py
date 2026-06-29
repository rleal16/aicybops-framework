#!/usr/bin/env python3
"""Run deploy/predict script suites and report pass/fail results."""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
ENV = {
    **os.environ,
    "PYTHONPATH": os.path.join(SCRIPTS_DIR, "..", "aicybops-lib", "src"),
    "PYTHONUNBUFFERED": "1",
}

@dataclass
class TestCase:
    id: str
    script: str
    args: List[str]
    expected_exits: Tuple[int, ...]
    description: str
    timeout: float = 600.0
    timeout_is_pass: bool = False
    output_checks: List[Tuple[str, bool]] = field(default_factory=list)


def build_suites(epochs: int) -> Tuple[List[TestCase], List[TestCase]]:
    E = str(epochs)

    suite_a = [
        TestCase(
            id="A1",
            script="deploy_model.py",
            args=["--help"],
            expected_exits=(0,),
            description="deploy_model.py --help",
            timeout=10,
            output_checks=[
                ("Train a DAM model", True),
                ("--no-optimize", True),   # optimisation is ON by default; flag disables it
                ("--no-evaluate", True),
                ("--verbose", True),
            ],
        ),
        TestCase(
            id="A2",
            script="deploy_model.py",
            args=["--no-optimize", "--epochs", E],
            expected_exits=(0,),
            description=f"deploy_model.py --no-optimize ({E} epochs)",
            output_checks=[
                ("Metrics:", True),
                ("Precision", True),
                ("Recall", True),
                ("F1 Score", True),
                ("[OK] Train", True),
                ("[OK] Evaluate", True),
            ],
        ),
        TestCase(
            id="A3",
            script="deploy_model.py",
            args=["--epochs", E, "--max-evals", "1"],
            expected_exits=(0,),
            description=f"deploy_model.py (optimised, 1 trial, {E} epochs)",
            timeout=900.0,
            output_checks=[
                ("Metrics:", True),
                ("Precision", True),
                ("[OK] Train", True),
            ],
        ),
        TestCase(
            id="A4",
            script="deploy_model.py",
            args=["--no-optimize", "--no-evaluate", "--epochs", E],
            expected_exits=(0,),
            description=f"deploy_model.py --no-optimize --no-evaluate ({E} epochs)",
            output_checks=[
                ("[OK] Train", True),
                ("[OK] Predict", True),
                ("4. Evaluate", False),   # evaluate section must NOT appear
            ],
        ),
        TestCase(
            id="A5",
            script="deploy_model.py",
            args=["--no-optimize", "--verbose", "--epochs", E],
            expected_exits=(0,),
            description=f"deploy_model.py --no-optimize --verbose ({E} epochs)",
            output_checks=[
                ("Request POST /train/", True),
                ("Response (full):", True),
                ("Metrics:", True),
            ],
        ),
        TestCase(
            id="A6",
            script="deploy_model.py",
            args=["--verbose", "--epochs", E, "--max-evals", "1"],
            expected_exits=(0,),
            description=f"deploy_model.py --verbose (optimised, 1 trial, {E} epochs)",
            timeout=900.0,
            output_checks=[
                ("Request POST /train/", True),
                ("Metrics:", True),
                ("Precision", True),
            ],
        ),
    ]

    suite_b = [
        TestCase(
            id="B1",
            script="predict_live.py",
            args=["--help"],
            expected_exits=(0,),
            description="predict_live.py --help",
            timeout=10,
            output_checks=[
                ("Score live data", True),
                ("--watch", True),
                ("--validate", True),
                ("--start", True),
            ],
        ),
        TestCase(
            id="B2",
            script="predict_live.py",
            args=[],
            expected_exits=(0, 1),
            description="predict_live.py (default)",
            output_checks=[
                ("Session log  : off", True),
                ("Service health: ok", True),
                ("Exit code  :", True),
            ],
        ),
        TestCase(
            id="B3",
            script="predict_live.py",
            args=["--validate"],
            expected_exits=(0, 1),
            description="predict_live.py --validate",
            output_checks=[
                ("session-log source:", True),
                ("Verdict :", True),
                ("Exit code  :", True),
            ],
        ),
        TestCase(
            id="B4",
            script="predict_live.py",
            args=["--start=-300s"],
            expected_exits=(0, 1),
            description="predict_live.py --start=-300s",
            output_checks=[
                ("-300s → now", True),
                ("error: argument", False),   # must NOT see argparse error
            ],
        ),
        TestCase(
            id="B5",
            script="predict_live.py",
            args=["--watch", "--interval", "30", "--validate"],
            expected_exits=(0,),
            description="predict_live.py --watch --interval 30 --validate (90s)",
            timeout=90.0,
            timeout_is_pass=True,
            output_checks=[
                ("Run #1", True),
                ("Run #2", True),
            ],
        ),
    ]

    return suite_a, suite_b


@dataclass
class TestResult:
    case: TestCase
    passed: bool
    exit_code: Optional[int]       # None on timeout/exception
    duration: float
    timed_out: bool
    error: Optional[str]           # exception message if subprocess raised
    output: str                    # captured stdout+stderr
    check_failures: List[str]      # which output checks failed


def run_test(case: TestCase) -> TestResult:
    cmd = [PYTHON, "-u", os.path.join(SCRIPTS_DIR, case.script)] + case.args
    t0 = time.monotonic()
    timed_out = False
    exit_code = None
    error = None
    output = ""

    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=ENV,
        ) as proc:
            try:
                stdout, _ = proc.communicate(timeout=case.timeout)
                exit_code = proc.returncode
                output = stdout or ""
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                stdout, _ = proc.communicate()
                output = stdout or ""
    except Exception as exc:
        error = str(exc)

    duration = time.monotonic() - t0

    # Evaluate exit code
    if timed_out:
        exit_ok = case.timeout_is_pass
    elif exit_code is not None:
        exit_ok = exit_code in case.expected_exits
    else:
        exit_ok = False

    # Evaluate output checks
    check_failures = []
    for substring, must_present in case.output_checks:
        found = substring in output
        if must_present and not found:
            check_failures.append(f"expected '{substring}' not found in output")
        elif not must_present and found:
            check_failures.append(f"unexpected '{substring}' found in output")

    passed = exit_ok and not check_failures and error is None

    return TestResult(
        case=case,
        passed=passed,
        exit_code=exit_code,
        duration=duration,
        timed_out=timed_out,
        error=error,
        output=output,
        check_failures=check_failures,
    )


W = 72


def _hr(char: str = "=") -> None:
    print(char * W)


def _section(title: str) -> None:
    print()
    _hr()
    print(f"  {title}")
    _hr()


def _print_result(r: TestResult) -> None:
    icon = "✅" if r.passed else "❌"
    if r.timed_out and r.case.timeout_is_pass:
        status = "PASS (timeout expected)"
    elif r.passed:
        status = "PASS"
    elif r.timed_out:
        status = "FAIL (unexpected timeout)"
    else:
        status = "FAIL"
    exit_str = str(r.exit_code) if r.exit_code is not None else ("timeout" if r.timed_out else "error")
    print(f"\n  {icon} [{r.case.id}] {r.case.description}  —  {status}  ({r.duration:.1f}s  exit={exit_str})")
    if r.check_failures:
        for cf in r.check_failures:
            print(f"      ⚠  Check failed: {cf}")
    if r.error:
        print(f"      ⚠  Exception: {r.error}")


def _print_output_block(r: TestResult, verbose: bool) -> None:
    """Print captured output. Always shown in full on failure; preview on pass."""
    if not r.output.strip():
        return
    lines = r.output.splitlines()
    if r.passed and not verbose:
        # Show a short preview: first 4 + last 4 lines
        if len(lines) > 10:
            preview = lines[:4] + [f"  ... ({len(lines) - 8} lines omitted) ..."] + lines[-4:]
        else:
            preview = lines
        print("  Output preview:")
        for line in preview:
            print(f"    {line}")
    else:
        print("  Full output:")
        for line in lines:
            print(f"    {line}")


def _inter_suite_wait(seconds: int) -> None:
    if seconds <= 0:
        return
    _section(f"Waiting {seconds}s between suites  (pass --wait 0 to skip)")
    deadline = time.time() + seconds
    interval = 60
    while True:
        remaining = int(deadline - time.time())
        if remaining <= 0:
            break
        print(f"  {remaining}s remaining...", end="\r", flush=True)
        time.sleep(min(interval, remaining))
    print(" " * 50, end="\r")  # clear countdown line
    print("  Done waiting.\n")


def _print_summary(results: List[TestResult]) -> None:
    _section("Summary")

    col_id   = 4
    col_desc = 46
    col_time = 8
    col_exit = 7
    col_res  = 28

    header = (
        f"  {'ID':<{col_id}}  {'Description':<{col_desc}}"
        f"  {'Time':>{col_time}}  {'Exit':>{col_exit}}  Result"
    )
    print(header)
    print("  " + "-" * (col_id + col_desc + col_time + col_exit + col_res + 8))

    passed_count = 0
    failed_count = 0
    for r in results:
        icon = "✅" if r.passed else "❌"
        if r.timed_out and r.case.timeout_is_pass:
            label = "PASS (timed out, expected)"
        elif r.passed:
            label = "PASS"
        elif r.timed_out:
            label = "FAIL (unexpected timeout)"
        else:
            label = "FAIL"

        if r.passed:
            passed_count += 1
        else:
            failed_count += 1

        exit_str = str(r.exit_code) if r.exit_code is not None else ("t/o" if r.timed_out else "err")
        desc = r.case.description
        if len(desc) > col_desc:
            desc = desc[:col_desc - 1] + "…"
        print(
            f"  {r.case.id:<{col_id}}  {desc:<{col_desc}}"
            f"  {r.duration:>{col_time - 1}.1f}s  {exit_str:>{col_exit}}  {icon} {label}"
        )
        for cf in r.check_failures:
            print(f"        ↳ {cf}")

    total = len(results)
    print()
    if failed_count == 0:
        print(f"  ✅  All {total} tests passed.")
    else:
        print(f"  ❌  {passed_count}/{total} passed — {failed_count} FAILED.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full option matrix for deploy_model.py and predict_live.py.",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=600,
        metavar="SECONDS",
        help="Seconds to wait between Suite A (deploy) and Suite B (predict). "
             "Pass 0 to skip. (default: 600)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for deploy_model.py tests. Lower = faster. (default: 10)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full output for all tests, not just failures.",
    )
    args = parser.parse_args()

    suite_a, suite_b = build_suites(args.epochs)

    _section("AICybOps — Script Test Runner")
    print(f"  Suite A (deploy_model.py)  : {len(suite_a)} tests")
    print(f"  Suite B (predict_live.py)  : {len(suite_b)} tests")
    print(f"  Inter-suite wait           : {args.wait}s")
    print(f"  Epochs (deploy tests)      : {args.epochs}")

    all_results: List[TestResult] = []

    # ── Suite A ──────────────────────────────────────────────────────────────
    _section("Suite A — deploy_model.py")
    for case in suite_a:
        print(f"\n{'─' * W}")
        print(f"  Running [{case.id}]: {case.description}")
        print(f"{'─' * W}")
        result = run_test(case)
        _print_result(result)
        _print_output_block(result, args.verbose)
        all_results.append(result)

    # ── Inter-suite wait ─────────────────────────────────────────────────────
    _inter_suite_wait(args.wait)

    # ── Suite B ──────────────────────────────────────────────────────────────
    _section("Suite B — predict_live.py")
    for case in suite_b:
        print(f"\n{'─' * W}")
        print(f"  Running [{case.id}]: {case.description}")
        print(f"{'─' * W}")
        result = run_test(case)
        _print_result(result)
        _print_output_block(result, args.verbose)
        all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    _print_summary(all_results)

    failed = [r for r in all_results if not r.passed]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
