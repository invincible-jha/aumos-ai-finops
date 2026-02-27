"""Abstract interfaces (Protocol classes) for the AumOS AI FinOps service.

All services depend on these interfaces, not concrete implementations.
This enables dependency injection and test doubles without coupling to
SQLAlchemy, Kafka, or external cost providers.
"""

import uuid
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

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


# ---------------------------------------------------------------------------
# Interfaces for domain-specific analytics adapters
# ---------------------------------------------------------------------------


@runtime_checkable
class ICostForecaster(Protocol):
    """Interface for AI cost trend analysis and forward-looking projections."""

    async def analyze_cost_trends(
        self,
        tenant_id: str,
        cost_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyse historical cost data to identify trends, velocity, and patterns.

        Args:
            tenant_id: Tenant for which to analyse costs.
            cost_history: List of period cost records with period_label and total_cost_usd.

        Returns:
            Dict with slope, r_squared, trend_direction, and period_summaries.
        """
        ...

    async def project_costs(
        self,
        tenant_id: str,
        cost_history: list[dict[str, Any]],
        projection_periods: int,
        method: str,
    ) -> dict[str, Any]:
        """Project future costs using linear or exponential extrapolation.

        Args:
            tenant_id: Tenant for which to project costs.
            cost_history: Historical cost records.
            projection_periods: Number of periods to project forward.
            method: Projection method: linear | exponential.

        Returns:
            Dict with projected_periods (list of period/cost pairs) and
            confidence_band.
        """
        ...

    async def detect_seasonal_patterns(
        self,
        tenant_id: str,
        cost_history: list[dict[str, Any]],
        period_type: str,
    ) -> dict[str, Any]:
        """Detect recurring seasonal patterns in cost data.

        Args:
            tenant_id: Tenant for which to detect patterns.
            cost_history: Historical cost records.
            period_type: Seasonality period: weekly | monthly | quarterly.

        Returns:
            Dict with seasonal_indices, peak_periods, and pattern_strength.
        """
        ...

    async def compare_budget_vs_actual(
        self,
        tenant_id: str,
        budget_periods: list[dict[str, Any]],
        actual_costs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare budgeted vs actual costs and compute variance metrics.

        Args:
            tenant_id: Tenant for budget comparison.
            budget_periods: List of periods with budget_usd per period.
            actual_costs: List of periods with actual_cost_usd per period.

        Returns:
            Dict with variance_by_period, cumulative_variance, and
            avg_variance_percent.
        """
        ...

    async def predict_budget_exhaustion(
        self,
        tenant_id: str,
        remaining_budget_usd: float,
        current_spend_rate_usd_per_day: float,
        spend_acceleration: float,
    ) -> dict[str, Any]:
        """Predict when the remaining budget will be exhausted.

        Args:
            tenant_id: Tenant for the prediction.
            remaining_budget_usd: Remaining budget in USD.
            current_spend_rate_usd_per_day: Current daily spend rate.
            spend_acceleration: Daily rate of change in spend (can be negative).

        Returns:
            Dict with exhaustion_days, exhaustion_date, and confidence.
        """
        ...

    async def detect_cost_anomalies(
        self,
        tenant_id: str,
        cost_history: list[dict[str, Any]],
        z_score_threshold: float,
    ) -> dict[str, Any]:
        """Detect anomalous cost spikes using Z-score analysis.

        Args:
            tenant_id: Tenant for anomaly detection.
            cost_history: Historical cost records.
            z_score_threshold: Z-score above which a period is flagged.

        Returns:
            Dict with anomalies (list of flagged periods) and anomaly_count.
        """
        ...

    async def generate_forecast_report(
        self,
        tenant_id: str,
        cost_history: list[dict[str, Any]],
        budget_usd: float | None,
        projection_periods: int,
    ) -> dict[str, Any]:
        """Generate a comprehensive cost forecast report.

        Args:
            tenant_id: Tenant for the report.
            cost_history: Historical cost records.
            budget_usd: Optional total budget for exhaustion analysis.
            projection_periods: Periods to project forward.

        Returns:
            Complete forecast report dict with trends, projections, anomalies,
            and budget health assessment.
        """
        ...


@runtime_checkable
class IInvoiceGenerator(Protocol):
    """Interface for per-tenant invoice compilation and reconciliation."""

    async def compile_tenant_invoice(
        self,
        tenant_id: str,
        billing_period_start: datetime,
        billing_period_end: datetime,
        cost_records: list[Any],
        token_usage_records: list[Any],
        tax_jurisdiction: str,
        payment_terms_days: int,
    ) -> dict[str, Any]:
        """Compile a complete invoice for a tenant covering a billing period.

        Args:
            tenant_id: Tenant for which to generate the invoice.
            billing_period_start: Inclusive start of the billing period.
            billing_period_end: Inclusive end of the billing period.
            cost_records: List of CostRecord objects or equivalent dicts.
            token_usage_records: List of TokenUsage objects or equivalent dicts.
            tax_jurisdiction: ISO country code for tax rate lookup.
            payment_terms_days: Net payment terms in days.

        Returns:
            Invoice dict with line_items, subtotal, tax, total, and metadata.
        """
        ...

    async def generate_pdf_invoice_data(
        self,
        invoice: dict[str, Any],
        company_details: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce a structured payload suitable for PDF invoice rendering.

        Args:
            invoice: Compiled invoice dict from compile_tenant_invoice.
            company_details: Billing company name, address, and logo URL.

        Returns:
            PDF-ready invoice data dict with formatted currency, layout hints,
            and QR code data for payment.
        """
        ...

    async def reconcile_with_provider_bill(
        self,
        tenant_id: str,
        internal_invoice: dict[str, Any],
        provider_bill: dict[str, Any],
        tolerance_percent: float,
    ) -> dict[str, Any]:
        """Reconcile the internal invoice against a provider bill.

        Args:
            tenant_id: Tenant for the reconciliation.
            internal_invoice: Invoice from compile_tenant_invoice.
            provider_bill: Provider bill dict with line_items and total.
            tolerance_percent: Acceptable variance percent before flagging.

        Returns:
            Reconciliation report with matched_items, unmatched_items,
            variance_usd, and reconciliation_status.
        """
        ...
