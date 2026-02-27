"""Chargeback and budget-limit service for the AumOS AI FinOps module.

Implements the FinOpsService class that backs the chargeback and budget-limit
API endpoints. Delegates all database operations to injected repository objects.

Key invariants:
  - All monetary values are USD.
  - Projected overrun is estimated as: daily_burn_rate * remaining_period_days - remaining_budget.
  - Budget alert logic fires when consumed_usd / limit_usd >= alert_threshold_pct / 100.
  - CSV generation is pure Python (no third-party library dependency).
"""
from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from aumos_ai_finops.api.schemas.finops import (
    BudgetLimitResponse,
    BudgetStatusResponse,
    ChargebackLineItem,
    ChargebackReportResponse,
    CostBreakdownResponse,
    CostLineItem,
)
from aumos_ai_finops.core.models.cost_allocation import BudgetLimit, CostAllocation
from aumos_ai_finops.settings import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Repository protocols (injected — not concrete implementations)
# ---------------------------------------------------------------------------


class ICostAllocationRepository:
    """Protocol for cost allocation repository operations."""

    async def list_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        group_by: str = "team",
    ) -> list[CostAllocation]:
        """List cost allocations for a tenant within a period."""
        ...

    async def list_all_for_report(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[CostAllocation]:
        """List all raw cost allocation rows for chargeback reporting."""
        ...


class IBudgetLimitRepository:
    """Protocol for budget limit repository operations."""

    async def create(self, limit: BudgetLimit) -> BudgetLimit:
        """Persist a new BudgetLimit."""
        ...

    async def get_by_team(
        self,
        tenant_id: str,
        team_id: str | None,
        period_type: str,
    ) -> BudgetLimit | None:
        """Find an active budget limit for a team (or tenant-wide if team_id is None)."""
        ...

    async def list_active(self, tenant_id: str) -> list[BudgetLimit]:
        """Return all active budget limits for a tenant."""
        ...


# ---------------------------------------------------------------------------
# FinOpsService
# ---------------------------------------------------------------------------


class FinOpsService:
    """Business logic for chargeback reports and budget limit management.

    Provides methods consumed by the finops API router. Delegates all I/O to
    injected repository objects to keep this class framework-agnostic.

    Args:
        allocation_repo: Repository for cost allocation data.
        budget_limit_repo: Repository for budget limit records.
        cost_repo: Repository for raw cost records (used for current-period spend).
        token_repo: Repository for token usage records (used for current-period spend).
        settings: Service configuration.
    """

    def __init__(
        self,
        allocation_repo: ICostAllocationRepository,
        budget_limit_repo: IBudgetLimitRepository,
        cost_repo: Any,
        token_repo: Any,
        settings: Settings,
    ) -> None:
        """Initialise FinOpsService with all required dependencies."""
        self._allocation_repo = allocation_repo
        self._budget_limit_repo = budget_limit_repo
        self._cost_repo = cost_repo
        self._token_repo = token_repo
        self._settings = settings

    # ------------------------------------------------------------------
    # Cost breakdown
    # ------------------------------------------------------------------

    async def get_cost_breakdown(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        group_by: str = "team",
    ) -> CostBreakdownResponse:
        """Get aggregated cost breakdown grouped by the specified dimension.

        Supported group_by values: team | project | model | service

        Args:
            tenant_id: The tenant whose costs to break down.
            period_start: Report window start (UTC).
            period_end: Report window end (UTC).
            group_by: Aggregation dimension.

        Returns:
            CostBreakdownResponse with line items sorted by cost descending.
        """
        allocations = await self._allocation_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            group_by=group_by,
        )

        # Aggregate allocations by the requested dimension
        aggregated: dict[str, dict[str, Any]] = {}
        for alloc in allocations:
            if group_by == "team":
                key = alloc.team_id
            elif group_by == "project":
                key = alloc.project_id
            elif group_by == "model":
                key = alloc.model_id or "unknown"
            elif group_by == "service":
                key = alloc.service
            else:
                key = alloc.team_id  # Fallback

            if key not in aggregated:
                aggregated[key] = {
                    "group_key": key,
                    "dimension": group_by,
                    "total_cost_usd": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "request_count": 0,
                    "period_start": period_start,
                    "period_end": period_end,
                }
            aggregated[key]["total_cost_usd"] += alloc.total_cost_usd
            aggregated[key]["total_input_tokens"] += alloc.total_input_tokens
            aggregated[key]["total_output_tokens"] += alloc.total_output_tokens

        line_items = [
            CostLineItem(**data)
            for data in sorted(aggregated.values(), key=lambda d: d["total_cost_usd"], reverse=True)
        ]
        total_cost_usd = sum(item.total_cost_usd for item in line_items)

        logger.info(
            "cost_breakdown_generated",
            tenant_id=tenant_id,
            group_by=group_by,
            line_item_count=len(line_items),
            total_cost_usd=total_cost_usd,
        )

        return CostBreakdownResponse(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            group_by=group_by,
            total_cost_usd=total_cost_usd,
            line_items=line_items,
        )

    # ------------------------------------------------------------------
    # Budget limits
    # ------------------------------------------------------------------

    async def create_budget(
        self,
        tenant_id: str,
        period_type: str,
        limit_usd: float,
        alert_threshold_pct: int = 80,
        hard_cap: bool = False,
        team_id: str | None = None,
    ) -> BudgetLimitResponse:
        """Create a new budget limit for a tenant or team.

        Args:
            tenant_id: The owning tenant.
            period_type: monthly | quarterly | annual
            limit_usd: Maximum spend in USD per period.
            alert_threshold_pct: Warning alert threshold (1–100).
            hard_cap: Whether to block requests that would exceed the limit.
            team_id: Optional team scope. None = tenant-wide limit.

        Returns:
            BudgetLimitResponse for the persisted record.
        """
        limit = BudgetLimit(
            tenant_id=tenant_id,
            team_id=team_id,
            period_type=period_type,
            limit_usd=limit_usd,
            alert_threshold_pct=alert_threshold_pct,
            hard_cap=hard_cap and self._settings.enable_hard_budget_caps,
            is_active=True,
        )
        persisted = await self._budget_limit_repo.create(limit)

        logger.info(
            "budget_limit_created",
            tenant_id=tenant_id,
            team_id=team_id,
            period_type=period_type,
            limit_usd=limit_usd,
            hard_cap=persisted.hard_cap,
        )

        return BudgetLimitResponse.model_validate(persisted)

    async def get_budget_status(
        self,
        tenant_id: str,
        team_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> BudgetStatusResponse:
        """Compute current utilisation status for a team's budget limit.

        Fetches the team's active budget limit, calculates current-period
        spend from cost and token records, and projects whether the team
        will overrun based on the daily burn rate.

        Args:
            tenant_id: The owning tenant.
            team_id: The team whose status to check.
            period_start: Start of the billing period.
            period_end: End of the billing period.

        Returns:
            BudgetStatusResponse with utilisation metrics.

        Raises:
            RuntimeError: If no active budget limit is configured for the team.
        """
        # Resolve period type from the period length
        period_days = max(1, (period_end - period_start).days)
        if period_days <= 31:
            period_type = "monthly"
        elif period_days <= 92:
            period_type = "quarterly"
        else:
            period_type = "annual"

        budget_limit = await self._budget_limit_repo.get_by_team(
            tenant_id=tenant_id,
            team_id=team_id,
            period_type=period_type,
        )
        if budget_limit is None:
            raise RuntimeError(
                f"No active {period_type} budget limit found for team '{team_id}' "
                f"in tenant '{tenant_id}'"
            )

        # Compute consumed spend: cost records + token costs
        consumed_usd = await self._cost_repo.sum_cost_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            scope="all",
        )
        token_spend = await self._token_repo.sum_cost_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )
        consumed_usd += token_spend

        remaining_usd = budget_limit.limit_usd - consumed_usd
        utilization_pct = (consumed_usd / budget_limit.limit_usd * 100) if budget_limit.limit_usd > 0 else 0.0
        alert_triggered = utilization_pct >= budget_limit.alert_threshold_pct

        # Project overrun: estimate based on daily burn rate
        now = datetime.now(timezone.utc)
        elapsed_days = max(1, (now - period_start).total_seconds() / 86_400)
        daily_burn = consumed_usd / elapsed_days
        remaining_period_days = max(0, (period_end - now).total_seconds() / 86_400)
        projected_remaining_cost = daily_burn * remaining_period_days
        projected_overrun_usd: float | None = None
        if projected_remaining_cost > remaining_usd and remaining_usd > 0:
            projected_overrun_usd = projected_remaining_cost - remaining_usd

        logger.debug(
            "budget_status_computed",
            tenant_id=tenant_id,
            team_id=team_id,
            consumed_usd=consumed_usd,
            utilization_pct=utilization_pct,
            alert_triggered=alert_triggered,
            projected_overrun_usd=projected_overrun_usd,
        )

        return BudgetStatusResponse(
            budget_limit_id=budget_limit.id,
            tenant_id=tenant_id,
            team_id=team_id,
            period_type=period_type,
            limit_usd=budget_limit.limit_usd,
            consumed_usd=consumed_usd,
            remaining_usd=remaining_usd,
            utilization_pct=utilization_pct,
            alert_threshold_pct=budget_limit.alert_threshold_pct,
            hard_cap=budget_limit.hard_cap,
            is_active=budget_limit.is_active,
            alert_triggered=alert_triggered,
            period_start=period_start,
            period_end=period_end,
            projected_overrun_usd=projected_overrun_usd,
        )

    # ------------------------------------------------------------------
    # Chargeback reporting
    # ------------------------------------------------------------------

    async def generate_chargeback_report(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        report_format: str = "json",
    ) -> ChargebackReportResponse:
        """Generate a full chargeback report for all teams in a period.

        Aggregates raw CostAllocation records into per-team/project line items
        and optionally formats as CSV.

        Args:
            tenant_id: The tenant to report on.
            period_start: Report period start (UTC).
            period_end: Report period end (UTC).
            report_format: Output format: json | csv | pdf (pdf returns json for now).

        Returns:
            ChargebackReportResponse with line items and optional csv_data.
        """
        allocations = await self._allocation_repo.list_all_for_report(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )

        line_items = [
            ChargebackLineItem(
                team_id=alloc.team_id,
                project_id=alloc.project_id,
                service=alloc.service,
                model_id=alloc.model_id,
                total_input_tokens=alloc.total_input_tokens,
                total_output_tokens=alloc.total_output_tokens,
                total_cost_usd=alloc.total_cost_usd,
                inference_minutes=alloc.inference_minutes,
                storage_gb_days=alloc.storage_gb_days,
            )
            for alloc in allocations
        ]

        line_items.sort(key=lambda item: item.total_cost_usd, reverse=True)
        total_cost_usd = sum(item.total_cost_usd for item in line_items)

        csv_data: str | None = None
        if report_format == "csv":
            csv_data = self._generate_csv(line_items)

        logger.info(
            "chargeback_report_generated",
            tenant_id=tenant_id,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            line_item_count=len(line_items),
            total_cost_usd=total_cost_usd,
            format=report_format,
        )

        return ChargebackReportResponse(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now(timezone.utc),
            total_cost_usd=total_cost_usd,
            line_items=line_items,
            format=report_format,
            csv_data=csv_data,
        )

    @staticmethod
    def _generate_csv(line_items: list[ChargebackLineItem]) -> str:
        """Render chargeback line items as a CSV string.

        Args:
            line_items: Line items to serialize.

        Returns:
            UTF-8 CSV string with a header row.
        """
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=[
                "team_id",
                "project_id",
                "service",
                "model_id",
                "total_input_tokens",
                "total_output_tokens",
                "total_cost_usd",
                "inference_minutes",
                "storage_gb_days",
            ],
        )
        writer.writeheader()
        for item in line_items:
            writer.writerow(item.model_dump())
        return buffer.getvalue()
