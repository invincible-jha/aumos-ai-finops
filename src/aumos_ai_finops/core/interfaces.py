"""Abstract interfaces (Protocol classes) for the AumOS AI FinOps service.

All services depend on these interfaces, not concrete implementations.
This enables dependency injection and test doubles without coupling to
SQLAlchemy, Kafka, or external cost providers.
"""

import uuid
from datetime import datetime
from typing import Protocol, runtime_checkable

from aumos_ai_finops.core.models import (
    Budget,
    BudgetAlert,
    CostRecord,
    ROICalculation,
    RoutingRecommendation,
    TokenUsage,
)


@runtime_checkable
class ICostRecordRepository(Protocol):
    """Repository interface for cost record persistence."""

    async def create(self, record: CostRecord) -> CostRecord:
        """Persist a new cost record."""
        ...

    async def get_by_id(self, record_id: uuid.UUID) -> CostRecord | None:
        """Retrieve a cost record by primary key."""
        ...

    async def list_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        resource_type: str | None = None,
    ) -> list[CostRecord]:
        """List cost records for a tenant within a time window, optionally filtered by resource type."""
        ...

    async def sum_cost_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        scope: str = "all",
    ) -> float:
        """Aggregate total cost for a tenant in a period. Scope filters to 'gpu', 'tokens', or 'all'."""
        ...

    async def list_gpu_costs(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        gpu_type: str | None = None,
    ) -> list[CostRecord]:
        """List GPU-specific cost records, optionally filtered by GPU model."""
        ...


@runtime_checkable
class ITokenUsageRepository(Protocol):
    """Repository interface for token usage persistence."""

    async def create(self, usage: TokenUsage) -> TokenUsage:
        """Persist a new token usage record."""
        ...

    async def get_by_id(self, usage_id: uuid.UUID) -> TokenUsage | None:
        """Retrieve a token usage record by primary key."""
        ...

    async def list_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[TokenUsage]:
        """List token usage records for a tenant within a time window."""
        ...

    async def aggregate_by_model(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[dict]:
        """Aggregate token usage grouped by model for a tenant period."""
        ...

    async def sum_cost_by_tenant_period(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> float:
        """Aggregate total token cost for a tenant in a period."""
        ...


@runtime_checkable
class IROICalculationRepository(Protocol):
    """Repository interface for ROI calculation persistence."""

    async def create(self, calculation: ROICalculation) -> ROICalculation:
        """Persist a new ROI calculation."""
        ...

    async def get_by_id(self, calculation_id: uuid.UUID) -> ROICalculation | None:
        """Retrieve an ROI calculation by primary key."""
        ...

    async def list_by_tenant(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ROICalculation]:
        """List ROI calculations for a tenant."""
        ...


@runtime_checkable
class IBudgetRepository(Protocol):
    """Repository interface for budget persistence."""

    async def create(self, budget: Budget) -> Budget:
        """Persist a new budget."""
        ...

    async def get_by_id(self, budget_id: uuid.UUID) -> Budget | None:
        """Retrieve a budget by primary key."""
        ...

    async def list_by_tenant(
        self,
        tenant_id: str,
        is_active: bool | None = None,
    ) -> list[Budget]:
        """List budgets for a tenant, optionally filtering by active status."""
        ...

    async def get_active_budgets_for_period(
        self,
        tenant_id: str,
        at_time: datetime,
    ) -> list[Budget]:
        """Get all active budgets covering a specific point in time."""
        ...


@runtime_checkable
class IBudgetAlertRepository(Protocol):
    """Repository interface for budget alert persistence."""

    async def create(self, alert: BudgetAlert) -> BudgetAlert:
        """Persist a new budget alert."""
        ...

    async def list_by_budget(
        self,
        budget_id: uuid.UUID,
        acknowledged: bool | None = None,
    ) -> list[BudgetAlert]:
        """List alerts for a budget, optionally filtering by acknowledgement status."""
        ...

    async def acknowledge(
        self,
        alert_id: uuid.UUID,
        acknowledged_by: uuid.UUID,
    ) -> BudgetAlert | None:
        """Mark a budget alert as acknowledged."""
        ...


@runtime_checkable
class IRoutingRecommendationRepository(Protocol):
    """Repository interface for routing recommendation persistence."""

    async def create(self, recommendation: RoutingRecommendation) -> RoutingRecommendation:
        """Persist a new routing recommendation."""
        ...

    async def get_latest_for_workload(
        self,
        tenant_id: str,
        workload_name: str,
        use_case: str,
    ) -> RoutingRecommendation | None:
        """Get the most recent routing recommendation for a workload+use_case pair."""
        ...


@runtime_checkable
class IOpenCostClient(Protocol):
    """Interface for the OpenCost HTTP API client."""

    async def get_allocation(
        self,
        namespace: str | None,
        window: str,
        aggregate: str,
    ) -> list[dict]:
        """Fetch cost allocation data from OpenCost for a time window."""
        ...

    async def health_check(self) -> bool:
        """Verify OpenCost API is reachable."""
        ...


@runtime_checkable
class IKubeCostClient(Protocol):
    """Interface for the KubeCost HTTP API client."""

    async def get_allocation(
        self,
        window: str,
        aggregate: str,
        namespace: str | None,
    ) -> list[dict]:
        """Fetch cost allocation data from KubeCost for a time window."""
        ...

    async def health_check(self) -> bool:
        """Verify KubeCost API is reachable."""
        ...


@runtime_checkable
class IFinOpsEventPublisher(Protocol):
    """Interface for publishing FinOps domain events to Kafka."""

    async def publish_cost_recorded(
        self,
        tenant_id: str,
        cost_record_id: str,
        resource_type: str,
        cost_usd: float,
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """Publish finops.cost_recorded event after a cost record is persisted."""
        ...

    async def publish_budget_exceeded(
        self,
        tenant_id: str,
        budget_id: str,
        budget_name: str,
        severity: str,
        actual_spend_usd: float,
        limit_usd: float,
        utilization_percent: float,
    ) -> None:
        """Publish finops.budget_exceeded event when a threshold is breached."""
        ...

    async def publish_roi_calculated(
        self,
        tenant_id: str,
        calculation_id: str,
        initiative_name: str,
        roi_percent: float,
        total_benefit_usd: float,
        total_ai_cost_usd: float,
    ) -> None:
        """Publish finops.roi_calculated event after an ROI calculation completes."""
        ...
