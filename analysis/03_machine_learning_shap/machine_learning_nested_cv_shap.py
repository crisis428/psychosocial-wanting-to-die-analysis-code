#!/usr/bin/env python3
"""Run the finalized machine-learning/SHAP workflow in its original cell order.

The workflow is split into readable step files only to make the public repository easier to browse.
All steps execute in one shared global namespace, preserving the finalized script logic.
"""
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent / "steps"
for step in sorted(STEP_DIR.glob("step_*.py")):
    print(f"\n=== Running {step.name} ===")
    exec(compile(step.read_text(encoding="utf-8"), str(step), "exec"), globals())
