"""Business logic services for the AumOS AI FinOps module.

Re-exports all service classes from both the original flat services.py and
the P0.2 FinOpsService in services/finops_service.py.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

# ---------------------------------------------------------------------------
# Load the sibling flat services.py (shadowed by this package directory).
# ---------------------------------------------------------------------------
_flat_path = pathlib.Path(__file__).parent.parent / "services.py"
_module_key = "aumos_ai_finops.core._services_base"

if _module_key not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_module_key, _flat_path)
    assert _spec is not None and _spec.loader is not None, (
        f"Could not locate flat services.py at {_flat_path}"
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_module_key] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_base = sys.modules[_module_key]

# Original services (re-exported for backward compatibility)
BudgetAlertService = _base.BudgetAlertService  # type: ignore[attr-defined]
CostCollectorService = _base.CostCollectorService  # type: ignore[attr-defined]
ROIEngineService = _base.ROIEngineService  # type: ignore[attr-defined]
RoutingOptimizerService = _base.RoutingOptimizerService  # type: ignore[attr-defined]
TokenTrackerService = _base.TokenTrackerService  # type: ignore[attr-defined]

# P0.2 chargeback service (from services/finops_service.py submodule)
from aumos_ai_finops.core.services.finops_service import FinOpsService  # noqa: E402

__all__ = [
    # Original services
    "BudgetAlertService",
    "CostCollectorService",
    "ROIEngineService",
    "RoutingOptimizerService",
    "TokenTrackerService",
    # P0.2 service
    "FinOpsService",
]
