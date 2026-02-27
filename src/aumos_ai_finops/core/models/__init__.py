"""Core ORM models for the AumOS AI FinOps service.

Re-exports all SQLAlchemy model classes so that imports can reference
the package root (aumos_ai_finops.core.models) rather than individual
submodules.

All model classes now live in ``core/models.py`` (the flat file that is
shadowed by this package directory). To provide backward-compatible imports
from ``aumos_ai_finops.core.models`` for the original classes, we load the
flat file once via importlib and register it in sys.modules so SQLAlchemy
sees a single class registry. The P0.2 chargeback models
(CostAllocation, BudgetLimit) are defined in the flat models.py as well
and re-exported here for convenience.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

# ---------------------------------------------------------------------------
# Load the sibling flat models.py by path and register it in sys.modules so
# that any subsequent ``import aumos_ai_finops.core._models_base`` picks up
# the same classes (single SQLAlchemy metadata registration).
# ---------------------------------------------------------------------------
_flat_path = pathlib.Path(__file__).parent.parent / "models.py"
_module_key = "aumos_ai_finops.core._models_base"

if _module_key not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_module_key, _flat_path)
    assert _spec is not None and _spec.loader is not None, (
        f"Could not locate flat models.py at {_flat_path}"
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_module_key] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_base = sys.modules[_module_key]

Budget = _base.Budget  # type: ignore[attr-defined]
BudgetAlert = _base.BudgetAlert  # type: ignore[attr-defined]
BudgetLimit = _base.BudgetLimit  # type: ignore[attr-defined]
CostAllocation = _base.CostAllocation  # type: ignore[attr-defined]
CostRecord = _base.CostRecord  # type: ignore[attr-defined]
ROICalculation = _base.ROICalculation  # type: ignore[attr-defined]
RoutingRecommendation = _base.RoutingRecommendation  # type: ignore[attr-defined]
TokenUsage = _base.TokenUsage  # type: ignore[attr-defined]

__all__ = [
    "Budget",
    "BudgetAlert",
    "BudgetLimit",
    "CostAllocation",
    "CostRecord",
    "ROICalculation",
    "RoutingRecommendation",
    "TokenUsage",
]
