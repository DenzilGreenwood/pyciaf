"""
Example runner for CIAF Agentic Execution Boundaries scenarios.

Run all example scenarios to demonstrate the complete system.

Created: 2026-03-28
"""

import sys

# Import scenarios
from healthcare_claims import run_healthcare_claims_scenario
from financial_approvals import run_financial_approvals_scenario
from production_changes import run_production_changes_scenario


def main():
    """Run all example scenarios."""
    scenarios = [
        ("Healthcare Claims Processing", run_healthcare_claims_scenario),
        ("Financial Payment Approvals", run_financial_approvals_scenario),
        ("Production Infrastructure Changes", run_production_changes_scenario),
    ]

    print("\n")
    print("=" * 80)
    print("CIAF AGENTIC EXECUTION BOUNDARIES - EXAMPLE SCENARIOS")
    print("=" * 80)
    print()
    print("Running all demonstration scenarios...")
    print()

    for i, (name, scenario_func) in enumerate(scenarios, 1):
        print(f"\n{'#' * 80}")
        print(f"# SCENARIO {i}: {name}")
        print(f"{'#' * 80}\n")

        try:
            scenario_func()
        except Exception as e:
            print(f"ERROR running scenario: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        print("\n")

    print("=" * 80)
    print("ALL SCENARIOS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
