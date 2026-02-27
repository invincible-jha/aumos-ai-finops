"""Pydantic schemas for the AumOS AI FinOps API.

Re-exports all request/response models for convenient importing.
This package re-exports schemas from both the original flat schemas.py and
the P0.2 chargeback schemas in schemas/finops.py.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

# ---------------------------------------------------------------------------
# Load the sibling flat schemas.py (shadowed by this package directory).
# ---------------------------------------------------------------------------
_flat_path = pathlib.Path(__file__).parent.parent / "schemas.py"
_module_key = "aumos_ai_finops.api._schemas_base"

if _module_key not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_module_key, _flat_path)
    assert _spec is not None and _spec.loader is not None, (
        f"Could not locate flat schemas.py at {_flat_path}"
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_module_key] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_base = sys.modules[_module_key]

# Original schemas (re-exported for backward compatibility)
BudgetAlertResponse = _base.BudgetAlertResponse  # type: ignore[attr-defined]
BudgetResponse = _base.BudgetResponse  # type: ignore[attr-defined]
CostRecordResponse = _base.CostRecordResponse  # type: ignore[attr-defined]
CreateBudgetRequest = _base.CreateBudgetRequest  # type: ignore[attr-defined]
DashboardSummaryResponse = _base.DashboardSummaryResponse  # type: ignore[attr-defined]
ModelTokenAggregateResponse = _base.ModelTokenAggregateResponse  # type: ignore[attr-defined]
ROICalculateRequest = _base.ROICalculateRequest  # type: ignore[attr-defined]
ROICalculationResponse = _base.ROICalculationResponse  # type: ignore[attr-defined]
RecordCostRequest = _base.RecordCostRequest  # type: ignore[attr-defined]
RecordTokenUsageRequest = _base.RecordTokenUsageRequest  # type: ignore[attr-defined]
RoutingOptimizeRequest = _base.RoutingOptimizeRequest  # type: ignore[attr-defined]
RoutingRecommendationResponse = _base.RoutingRecommendationResponse  # type: ignore[attr-defined]
SyncOpenCostRequest = _base.SyncOpenCostRequest  # type: ignore[attr-defined]
TokenUsageResponse = _base.TokenUsageResponse  # type: ignore[attr-defined]
CostQueryParams = _base.CostQueryParams  # type: ignore[attr-defined]

# P0.2 chargeback schemas (from schemas/finops.py submodule)
from aumos_ai_finops.api.schemas.finops import (  # noqa: E402
    BudgetCreateRequest,
    BudgetLimitResponse,
    BudgetStatusResponse,
    ChargebackLineItem,
    ChargebackReportResponse,
    CostBreakdownResponse,
    CostLineItem,
)

__all__ = [
    # Original schemas
    "BudgetAlertResponse",
    "BudgetResponse",
    "CostRecordResponse",
    "CostQueryParams",
    "CreateBudgetRequest",
    "DashboardSummaryResponse",
    "ModelTokenAggregateResponse",
    "ROICalculateRequest",
    "ROICalculationResponse",
    "RecordCostRequest",
    "RecordTokenUsageRequest",
    "RoutingOptimizeRequest",
    "RoutingRecommendationResponse",
    "SyncOpenCostRequest",
    "TokenUsageResponse",
    # P0.2 chargeback schemas
    "BudgetCreateRequest",
    "BudgetLimitResponse",
    "BudgetStatusResponse",
    "ChargebackLineItem",
    "ChargebackReportResponse",
    "CostBreakdownResponse",
    "CostLineItem",
]
