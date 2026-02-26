"""SQLAlchemy repositories for the AumOS AI FinOps service.

All repositories extend BaseRepository from aumos-common and implement
the interfaces defined in core/interfaces.py. RLS tenant isolation is
enforced automatically via the get_db_session dependency.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import BaseRepository
from aumos_common.observability import get_logger

from aumos_ai_finops.core.models import (
    Budget,
    BudgetAlert,
    CostRecord,
    ROICalculation,
    RoutingRecommendation,
    TokenUsage,
)

logger = get_logger(__name__)


class CostRecordRepository(BaseRepository[CostRecord]):
    """Repository for fin_cost_records — GPU/CPU/storage cost allocations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an async database session."""
        super().__init__(session, CostRecord)

    async def list_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        resource_type: str | None = None,
    ) -> list[CostRecord]:
        """List cost records for a tenant within a time window.

        Args:
            tenant_id: Tenant filter (RLS also enforces this).
            period_start: Window start (inclusive).
            period_end: Window end (inclusive).
            resource_type: Optional filter: gpu | cpu | storage | network | memory

        Returns:
            List of matching CostRecord objects ordered by period_start descending.
        """
        query = (
            select(CostRecord)
            .where(
                CostRecord.tenant_id == tenant_id,
                CostRecord.period_start >= period_start,
                CostRecord.period_end <= period_end,
            )
            .order_by(CostRecord.period_start.desc())
        )
        if resource_type is not None:
            query = query.where(CostRecord.resource_type == resource_type)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def sum_cost_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        scope: str = "all",
    ) -> float:
        """Aggregate total cost for a tenant in a period.

        Args:
            tenant_id: Tenant filter.
            period_start: Window start.
            period_end: Window end.
            scope: 'gpu' filters to gpu resource_type, 'all' includes everything.

        Returns:
            Total cost in USD (0.0 if no records).
        """
        query = select(func.coalesce(func.sum(CostRecord.cost_usd), 0.0)).where(
            CostRecord.tenant_id == tenant_id,
            CostRecord.period_start >= period_start,
            CostRecord.period_end <= period_end,
        )
        if scope == "gpu":
            query = query.where(CostRecord.resource_type == "gpu")

        result = await self._session.execute(query)
        return float(result.scalar() or 0.0)

    async def list_gpu_costs(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        gpu_type: str | None = None,
    ) -> list[CostRecord]:
        """List GPU cost records for a tenant, optionally filtered by GPU model.

        Args:
            tenant_id: Tenant filter.
            period_start: Window start.
            period_end: Window end.
            gpu_type: Optional GPU model filter: a100 | h100 | t4

        Returns:
            List of GPU CostRecord objects.
        """
        query = (
            select(CostRecord)
            .where(
                CostRecord.tenant_id == tenant_id,
                CostRecord.resource_type == "gpu",
                CostRecord.period_start >= period_start,
                CostRecord.period_end <= period_end,
            )
            .order_by(CostRecord.period_start.desc())
        )
        if gpu_type is not None:
            query = query.where(CostRecord.gpu_type == gpu_type)

        result = await self._session.execute(query)
        return list(result.scalars().all())


class TokenUsageRepository(BaseRepository[TokenUsage]):
    """Repository for fin_token_usage — per-model token consumption records."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an async database session."""
        super().__init__(session, TokenUsage)

    async def list_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[TokenUsage]:
        """List token usage records for a tenant in a time window.

        Args:
            tenant_id: Tenant filter.
            period_start: Window start (inclusive).
            period_end: Window end (inclusive).

        Returns:
            List of TokenUsage objects ordered by period_start descending.
        """
        query = (
            select(TokenUsage)
            .where(
                TokenUsage.tenant_id == tenant_id,
                TokenUsage.period_start >= period_start,
                TokenUsage.period_end <= period_end,
            )
            .order_by(TokenUsage.period_start.desc())
        )
        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def aggregate_by_model(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[dict[str, Any]]:
        """Aggregate token usage grouped by model_name for a tenant period.

        Args:
            tenant_id: Tenant filter.
            period_start: Window start.
            period_end: Window end.

        Returns:
            List of dicts: {model_name, provider, tier, total_tokens,
                            prompt_tokens, completion_tokens, total_cost_usd, request_count}
        """
        query = (
            select(
                TokenUsage.model_name,
                TokenUsage.model_provider,
                TokenUsage.model_tier,
                func.sum(TokenUsage.prompt_tokens).label("prompt_tokens"),
                func.sum(TokenUsage.completion_tokens).label("completion_tokens"),
                func.sum(TokenUsage.total_tokens).label("total_tokens"),
                func.sum(TokenUsage.total_cost_usd).label("total_cost_usd"),
                func.sum(TokenUsage.request_count).label("request_count"),
            )
            .where(
                TokenUsage.tenant_id == tenant_id,
                TokenUsage.period_start >= period_start,
                TokenUsage.period_end <= period_end,
            )
            .group_by(TokenUsage.model_name, TokenUsage.model_provider, TokenUsage.model_tier)
            .order_by(func.sum(TokenUsage.total_cost_usd).desc())
        )
        result = await self._session.execute(query)
        rows = result.all()
        return [
            {
                "model_name": row.model_name,
                "model_provider": row.model_provider,
                "model_tier": row.model_tier,
                "prompt_tokens": int(row.prompt_tokens),
                "completion_tokens": int(row.completion_tokens),
                "total_tokens": int(row.total_tokens),
                "total_cost_usd": float(row.total_cost_usd),
                "request_count": int(row.request_count),
            }
            for row in rows
        ]

    async def sum_cost_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> float:
        """Aggregate total token cost for a tenant in a period.

        Args:
            tenant_id: Tenant filter.
            period_start: Window start.
            period_end: Window end.

        Returns:
            Total token cost in USD.
        """
        query = select(func.coalesce(func.sum(TokenUsage.total_cost_usd), 0.0)).where(
            TokenUsage.tenant_id == tenant_id,
            TokenUsage.period_start >= period_start,
            TokenUsage.period_end <= period_end,
        )
        result = await self._session.execute(query)
        return float(result.scalar() or 0.0)


class ROICalculationRepository(BaseRepository[ROICalculation]):
    """Repository for fin_roi_calculations — multi-touch attribution ROI records."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an async database session."""
        super().__init__(session, ROICalculation)

    async def list_by_tenant(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ROICalculation]:
        """List ROI calculations for a tenant.

        Args:
            tenant_id: Tenant filter.
            status: Optional status filter: completed | draft | archived
            limit: Maximum number of results.
            offset: Number of records to skip.

        Returns:
            List of ROICalculation objects ordered by created_at descending.
        """
        query = (
            select(ROICalculation)
            .where(ROICalculation.tenant_id == tenant_id)
            .order_by(ROICalculation.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if status is not None:
            query = query.where(ROICalculation.status == status)

        result = await self._session.execute(query)
        return list(result.scalars().all())


class BudgetRepository(BaseRepository[Budget]):
    """Repository for fin_budgets — per-tenant budget thresholds."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an async database session."""
        super().__init__(session, Budget)

    async def list_by_tenant(
        self,
        tenant_id: str,
        is_active: bool | None = None,
    ) -> list[Budget]:
        """List budgets for a tenant.

        Args:
            tenant_id: Tenant filter.
            is_active: Optional filter for active/inactive budgets.

        Returns:
            List of Budget objects ordered by period_start descending.
        """
        query = (
            select(Budget)
            .where(Budget.tenant_id == tenant_id)
            .order_by(Budget.period_start.desc())
        )
        if is_active is not None:
            query = query.where(Budget.is_active == is_active)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def get_active_budgets_for_period(
        self,
        tenant_id: str,
        at_time: datetime,
    ) -> list[Budget]:
        """Get all active budgets covering a specific point in time.

        Args:
            tenant_id: Tenant filter.
            at_time: The point in time the budget must cover.

        Returns:
            List of active Budget objects where period_start <= at_time <= period_end.
        """
        query = (
            select(Budget)
            .where(
                Budget.tenant_id == tenant_id,
                Budget.is_active.is_(True),
                Budget.period_start <= at_time,
                Budget.period_end >= at_time,
            )
            .order_by(Budget.created_at.asc())
        )
        result = await self._session.execute(query)
        return list(result.scalars().all())


class BudgetAlertRepository(BaseRepository[BudgetAlert]):
    """Repository for fin_budget_alerts — triggered budget threshold alerts."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an async database session."""
        super().__init__(session, BudgetAlert)

    async def list_by_budget(
        self,
        budget_id: uuid.UUID,
        acknowledged: bool | None = None,
    ) -> list[BudgetAlert]:
        """List alerts for a specific budget.

        Args:
            budget_id: Parent budget ID.
            acknowledged: Optional filter for acknowledgement status.

        Returns:
            List of BudgetAlert objects ordered by created_at descending.
        """
        query = (
            select(BudgetAlert)
            .where(BudgetAlert.budget_id == budget_id)
            .order_by(BudgetAlert.created_at.desc())
        )
        if acknowledged is not None:
            query = query.where(BudgetAlert.acknowledged == acknowledged)

        result = await self._session.execute(query)
        return list(result.scalars().all())

    async def acknowledge(
        self,
        alert_id: uuid.UUID,
        acknowledged_by: uuid.UUID,
    ) -> BudgetAlert | None:
        """Mark a budget alert as acknowledged.

        Args:
            alert_id: The alert to acknowledge.
            acknowledged_by: UUID of the user acknowledging the alert.

        Returns:
            Updated BudgetAlert, or None if not found.
        """
        from datetime import timezone

        alert = await self.get_by_id(alert_id)
        if alert is None:
            return None

        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now(timezone.utc)
        await self._session.flush()
        return alert


class RoutingRecommendationRepository(BaseRepository[RoutingRecommendation]):
    """Repository for fin_routing_recommendations — cost-optimized routing suggestions."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with an async database session."""
        super().__init__(session, RoutingRecommendation)

    async def get_latest_for_workload(
        self,
        tenant_id: str,
        workload_name: str,
        use_case: str,
    ) -> RoutingRecommendation | None:
        """Get the most recent routing recommendation for a workload+use_case pair.

        Args:
            tenant_id: Tenant filter.
            workload_name: Service or agent name.
            use_case: Business use case label.

        Returns:
            Most recent RoutingRecommendation, or None if no history exists.
        """
        query = (
            select(RoutingRecommendation)
            .where(
                RoutingRecommendation.tenant_id == tenant_id,
                RoutingRecommendation.workload_name == workload_name,
                RoutingRecommendation.use_case == use_case,
            )
            .order_by(RoutingRecommendation.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(query)
        return result.scalars().first()
