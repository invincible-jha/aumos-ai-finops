"""FastAPI router for the AumOS AI FinOps chargeback and budget-limit API.

All routes are thin — validate inputs, call FinOpsService, return typed
Pydantic response models. No business logic belongs here.

Endpoints:
  GET  /api/v1/finops/costs/breakdown
        Query params: tenant_id, period_start, period_end, group_by=[team|project|model|service]

  POST /api/v1/finops/budgets
        Create a new budget limit for a team or tenant.

  GET  /api/v1/finops/budgets/{team_id}/status
        Query params: period_start, period_end

  GET  /api/v1/finops/costs/chargeback-report
        Query params: period_start, period_end, format=[json|csv|pdf]
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session

from aumos_ai_finops.adapters.repositories import (
    CostRecordRepository,
    TokenUsageRepository,
)
from aumos_ai_finops.api.schemas.finops import (
    BudgetCreateRequest,
    BudgetLimitResponse,
    BudgetStatusResponse,
    ChargebackReportResponse,
    CostBreakdownResponse,
)
from aumos_ai_finops.core.services.finops_service import FinOpsService
from aumos_ai_finops.settings import Settings

router = APIRouter(prefix="/finops", tags=["finops-chargeback"])
settings = Settings()


# ---------------------------------------------------------------------------
# Stub repositories for CostAllocation and BudgetLimit
# (full SQLAlchemy implementations belong in adapters/repositories.py —
#  these inline stubs satisfy the service protocol for import verification)
# ---------------------------------------------------------------------------


class _InMemoryCostAllocationRepo:
    """In-memory stub for CostAllocation repository (for import/test verification)."""

    async def list_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        group_by: str = "team",
    ) -> list:
        return []

    async def list_all_for_report(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list:
        return []


class _InMemoryBudgetLimitRepo:
    """In-memory stub for BudgetLimit repository (for import/test verification)."""

    async def create(self, limit: object) -> object:
        return limit

    async def get_by_team(
        self,
        tenant_id: str,
        team_id: str | None,
        period_type: str,
    ) -> None:
        return None

    async def list_active(self, tenant_id: str) -> list:
        return []


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def _get_finops_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> FinOpsService:
    """Build FinOpsService with all required dependencies."""
    return FinOpsService(
        allocation_repo=_InMemoryCostAllocationRepo(),
        budget_limit_repo=_InMemoryBudgetLimitRepo(),
        cost_repo=CostRecordRepository(session),
        token_repo=TokenUsageRepository(session),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Cost breakdown endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/costs/breakdown",
    response_model=CostBreakdownResponse,
    summary="Get AI cost breakdown grouped by team, project, model, or service",
)
async def get_cost_breakdown(
    period_start: Annotated[datetime, Query(description="Report period start (UTC)")],
    period_end: Annotated[datetime, Query(description="Report period end (UTC)")],
    group_by: Annotated[
        str,
        Query(description="Grouping dimension: team | project | model | service"),
    ] = "team",
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[FinOpsService, Depends(_get_finops_service)] = ...,
) -> CostBreakdownResponse:
    """Get AI cost breakdown for the current tenant grouped by the specified dimension.

    Aggregates all cost allocation records within the time window and groups
    them by team, project, model, or service. Results are sorted by total
    cost descending to surface the highest-spend categories first.
    """
    return await service.get_cost_breakdown(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=period_end,
        group_by=group_by,
    )


# ---------------------------------------------------------------------------
# Budget limit endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/budgets",
    response_model=BudgetLimitResponse,
    status_code=201,
    summary="Create a budget limit for a team or tenant",
)
async def create_budget(
    request: BudgetCreateRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[FinOpsService, Depends(_get_finops_service)] = ...,
) -> BudgetLimitResponse:
    """Create a new spending limit for a team or the entire tenant.

    When hard_cap=True and AUMOS_FINOPS_ENABLE_HARD_BUDGET_CAPS is set in the
    service environment, requests that would cause spend to exceed the limit
    are blocked with HTTP 429.
    """
    return await service.create_budget(
        tenant_id=tenant.tenant_id,
        team_id=request.team_id,
        period_type=request.period_type,
        limit_usd=request.limit_usd,
        alert_threshold_pct=request.alert_threshold_pct,
        hard_cap=request.hard_cap,
    )


@router.get(
    "/budgets/{team_id}/status",
    response_model=BudgetStatusResponse,
    summary="Get current budget utilisation status for a team",
)
async def get_budget_status(
    team_id: str,
    period_start: Annotated[datetime, Query(description="Billing period start (UTC)")],
    period_end: Annotated[datetime, Query(description="Billing period end (UTC)")],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[FinOpsService, Depends(_get_finops_service)] = ...,
) -> BudgetStatusResponse:
    """Get current budget utilisation for a specific team.

    Returns consumed spend, remaining budget, utilisation percentage, and a
    projected overrun estimate based on the team's daily burn rate.
    """
    return await service.get_budget_status(
        tenant_id=tenant.tenant_id,
        team_id=team_id,
        period_start=period_start,
        period_end=period_end,
    )


# ---------------------------------------------------------------------------
# Chargeback report endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/costs/chargeback-report",
    response_model=ChargebackReportResponse,
    summary="Generate a chargeback report for all teams in a billing period",
)
async def get_chargeback_report(
    period_start: Annotated[datetime, Query(description="Report period start (UTC)")],
    period_end: Annotated[datetime, Query(description="Report period end (UTC)")],
    format: Annotated[
        str,
        Query(description="Output format: json | csv | pdf"),
    ] = "json",
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[FinOpsService, Depends(_get_finops_service)] = ...,
) -> ChargebackReportResponse:
    """Generate a full chargeback report for all teams within a billing period.

    Returns per-team, per-project cost allocations including token counts,
    inference minutes, and storage. When format=csv, the response also includes
    a csv_data field with the full CSV string.
    """
    return await service.generate_chargeback_report(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=period_end,
        report_format=format,
    )
