"""Business logic services for the AumOS AI FinOps service.

All services depend on repository and adapter interfaces (not concrete
implementations) and receive dependencies via constructor injection.
No framework code (FastAPI, SQLAlchemy) belongs here.

Key invariants:
- CostCollectorService: Pulls cost data from OpenCost/KubeCost and persists fin_cost_records.
- TokenTrackerService: Records token usage and computes per-model cost analytics.
- ROIEngineService: Computes multi-touch attribution ROI using the standard formula.
- BudgetAlertService: Evaluates budget thresholds and triggers alerts when exceeded.
- RoutingOptimizerService: Scores model candidates and recommends cost-optimal routing.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError
from aumos_common.observability import get_logger

from aumos_ai_finops.core.interfaces import (
    IBudgetAlertRepository,
    IBudgetRepository,
    ICostForecaster,
    ICostRecordRepository,
    IFinOpsEventPublisher,
    IInvoiceGenerator,
    IKubeCostClient,
    IOpenCostClient,
    IROICalculationRepository,
    IRoutingRecommendationRepository,
    ITokenUsageRepository,
)
from aumos_ai_finops.core.models import (
    Budget,
    BudgetAlert,
    CostRecord,
    ROICalculation,
    RoutingRecommendation,
    TokenUsage,
)
from aumos_ai_finops.settings import Settings

logger = get_logger(__name__)


class CostCollectorService:
    """Collect and persist GPU/infrastructure cost data from external providers.

    Pulls allocation data from OpenCost and/or KubeCost, normalizes it to
    fin_cost_records, and publishes finops.cost_recorded events.
    """

    def __init__(
        self,
        cost_repo: ICostRecordRepository,
        opencost_client: IOpenCostClient | None,
        kubecost_client: IKubeCostClient | None,
        event_publisher: IFinOpsEventPublisher,
        settings: Settings,
    ) -> None:
        """Initialize CostCollectorService with required dependencies."""
        self._cost_repo = cost_repo
        self._opencost_client = opencost_client
        self._kubecost_client = kubecost_client
        self._publisher = event_publisher
        self._settings = settings

    async def record_cost(
        self,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        cost_usd: float,
        period_start: datetime,
        period_end: datetime,
        gpu_type: str | None = None,
        workload_name: str | None = None,
        namespace: str | None = None,
        model_id: uuid.UUID | None = None,
        on_demand_cost_usd: float = 0.0,
        efficiency_rate: float = 1.0,
        source: str = "manual",
        raw_metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Record a cost allocation and publish a cost_recorded event.

        Args:
            tenant_id: The tenant this cost belongs to.
            resource_type: gpu | cpu | storage | network | memory
            resource_id: Provider-specific resource identifier.
            cost_usd: Total cost in USD.
            period_start: Start of the billing window (UTC).
            period_end: End of the billing window (UTC).
            gpu_type: GPU model (a100 | h100 | t4) for GPU resources.
            workload_name: Kubernetes workload name.
            namespace: Kubernetes namespace.
            model_id: Associated model from aumos-model-registry.
            on_demand_cost_usd: On-demand equivalent cost for savings calc.
            efficiency_rate: Resource utilization rate (0.0–1.0).
            source: Data source (opencost | kubecost | manual | provider_api).
            raw_metadata: Raw provider response for audit purposes.

        Returns:
            The persisted CostRecord.
        """
        record = CostRecord(
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            cost_usd=cost_usd,
            period_start=period_start,
            period_end=period_end,
            gpu_type=gpu_type,
            workload_name=workload_name,
            namespace=namespace,
            model_id=model_id,
            on_demand_cost_usd=on_demand_cost_usd if on_demand_cost_usd > 0 else cost_usd,
            efficiency_rate=efficiency_rate,
            source=source,
            raw_metadata=raw_metadata or {},
        )

        persisted = await self._cost_repo.create(record)

        await self._publisher.publish_cost_recorded(
            tenant_id=tenant_id,
            cost_record_id=str(persisted.id),
            resource_type=resource_type,
            cost_usd=cost_usd,
            period_start=period_start,
            period_end=period_end,
        )

        logger.info(
            "cost_record_created",
            tenant_id=tenant_id,
            record_id=str(persisted.id),
            resource_type=resource_type,
            cost_usd=cost_usd,
            source=source,
        )

        return persisted

    async def sync_from_opencost(
        self,
        tenant_id: str,
        namespace: str | None,
        window: str,
    ) -> list[CostRecord]:
        """Pull cost data from OpenCost and persist as cost records.

        Args:
            tenant_id: Target tenant for cost allocation.
            namespace: Kubernetes namespace to filter (None = all namespaces).
            window: OpenCost time window string (e.g., '24h', '7d', '2024-01-01T00:00:00Z,2024-01-31T23:59:59Z').

        Returns:
            List of persisted CostRecord objects.

        Raises:
            RuntimeError: If OpenCost is not configured/enabled.
        """
        if self._opencost_client is None or not self._settings.opencost_enabled:
            raise RuntimeError("OpenCost integration is not enabled")

        allocations = await self._opencost_client.get_allocation(
            namespace=namespace,
            window=window,
            aggregate="namespace",
        )

        records: list[CostRecord] = []
        for alloc in allocations:
            try:
                record = await self.record_cost(
                    tenant_id=tenant_id,
                    resource_type=alloc.get("resourceType", "cpu"),
                    resource_id=alloc.get("name", "unknown"),
                    cost_usd=float(alloc.get("totalCost", 0.0)),
                    period_start=datetime.fromisoformat(alloc.get("start", "")),
                    period_end=datetime.fromisoformat(alloc.get("end", "")),
                    namespace=alloc.get("namespace"),
                    efficiency_rate=float(alloc.get("cpuEfficiency", 1.0)),
                    source="opencost",
                    raw_metadata=alloc,
                )
                records.append(record)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "opencost_allocation_parse_error",
                    tenant_id=tenant_id,
                    error=str(exc),
                    allocation=alloc,
                )

        logger.info(
            "opencost_sync_completed",
            tenant_id=tenant_id,
            window=window,
            records_created=len(records),
        )
        return records

    async def get_cost_breakdown(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        resource_type: str | None = None,
    ) -> list[CostRecord]:
        """Get cost records for a tenant in a period, optionally filtered by resource type."""
        return await self._cost_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            resource_type=resource_type,
        )

    async def get_gpu_costs(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        gpu_type: str | None = None,
    ) -> list[CostRecord]:
        """Get GPU-specific cost records for a tenant."""
        return await self._cost_repo.list_gpu_costs(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            gpu_type=gpu_type,
        )


class TokenTrackerService:
    """Track LLM token consumption and compute per-model cost analytics.

    Receives token usage data (typically from Kafka events published by
    aumos-llm-serving) and provides analytics including per-model breakdowns
    and cost trend data for the executive dashboard.
    """

    def __init__(
        self,
        token_repo: ITokenUsageRepository,
        settings: Settings,
    ) -> None:
        """Initialize TokenTrackerService with required dependencies."""
        self._token_repo = token_repo
        self._settings = settings

    def _compute_token_cost(
        self,
        model_tier: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> tuple[float, float]:
        """Compute prompt and completion token costs based on model tier.

        Args:
            model_tier: tier1 (premium) | tier2 (mid) | tier3 (economy)
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Tuple of (prompt_cost_usd, completion_cost_usd).
        """
        tier_costs: dict[str, float] = {
            "tier1": self._settings.token_cost_per_million_input_tier1,
            "tier2": self._settings.token_cost_per_million_input_tier2,
            "tier3": self._settings.token_cost_per_million_input_tier3,
        }
        cost_per_million = tier_costs.get(model_tier, self._settings.token_cost_per_million_input_tier2)
        # Output tokens typically cost 3x input tokens for most providers
        completion_cost_per_million = cost_per_million * 3.0

        prompt_cost_usd = (prompt_tokens / 1_000_000) * cost_per_million
        completion_cost_usd = (completion_tokens / 1_000_000) * completion_cost_per_million

        return prompt_cost_usd, completion_cost_usd

    async def record_token_usage(
        self,
        tenant_id: str,
        model_id: uuid.UUID,
        model_name: str,
        model_provider: str,
        model_tier: str,
        period_start: datetime,
        period_end: datetime,
        prompt_tokens: int,
        completion_tokens: int,
        request_count: int = 1,
        workload_name: str | None = None,
        use_case: str | None = None,
    ) -> TokenUsage:
        """Record token consumption and persist it.

        Args:
            tenant_id: The tenant consuming tokens.
            model_id: Model registry UUID.
            model_name: Human-readable model name.
            model_provider: Provider name (openai | anthropic | etc.).
            model_tier: Cost tier (tier1 | tier2 | tier3).
            period_start: Start of the aggregation window.
            period_end: End of the aggregation window.
            prompt_tokens: Total input tokens consumed.
            completion_tokens: Total output tokens generated.
            request_count: Number of model calls aggregated.
            workload_name: Service or agent making the calls.
            use_case: Business use case label.

        Returns:
            The persisted TokenUsage record.
        """
        prompt_cost_usd, completion_cost_usd = self._compute_token_cost(
            model_tier=model_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_tokens = prompt_tokens + completion_tokens
        total_cost_usd = prompt_cost_usd + completion_cost_usd

        usage = TokenUsage(
            tenant_id=tenant_id,
            model_id=model_id,
            model_name=model_name,
            model_provider=model_provider,
            model_tier=model_tier,
            period_start=period_start,
            period_end=period_end,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_count=request_count,
            prompt_cost_usd=prompt_cost_usd,
            completion_cost_usd=completion_cost_usd,
            total_cost_usd=total_cost_usd,
            workload_name=workload_name,
            use_case=use_case,
        )

        persisted = await self._token_repo.create(usage)

        logger.info(
            "token_usage_recorded",
            tenant_id=tenant_id,
            model_name=model_name,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            request_count=request_count,
        )

        return persisted

    async def get_token_analytics(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[TokenUsage]:
        """Get token usage records for a tenant in a period."""
        return await self._token_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )

    async def get_usage_by_model(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[dict[str, Any]]:
        """Get token usage aggregated by model for a tenant period.

        Returns a list of dicts with model-level totals and costs.
        """
        return await self._token_repo.aggregate_by_model(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )


class ROIEngineService:
    """Compute multi-touch attribution ROI for AI initiatives.

    Formula:
        ROI = (productivity_gain + quality_improvement + risk_avoidance - ai_cost)
              / ai_cost * 100

    Where:
        ai_cost = gpu_cost + token_cost + infra_cost
        (pulled from fin_cost_records and fin_token_usage for the period)
    """

    def __init__(
        self,
        roi_repo: IROICalculationRepository,
        cost_repo: ICostRecordRepository,
        token_repo: ITokenUsageRepository,
        event_publisher: IFinOpsEventPublisher,
        settings: Settings,
    ) -> None:
        """Initialize ROIEngineService with required dependencies."""
        self._roi_repo = roi_repo
        self._cost_repo = cost_repo
        self._token_repo = token_repo
        self._publisher = event_publisher
        self._settings = settings

    def _compute_payback_period(
        self,
        total_benefit_usd: float,
        total_ai_cost_usd: float,
        period_days: int,
    ) -> int | None:
        """Estimate payback period in days.

        Args:
            total_benefit_usd: Total benefit in the measurement period.
            total_ai_cost_usd: Total AI cost in the measurement period.
            period_days: Length of the measurement period in days.

        Returns:
            Estimated days to break even, or None if cost exceeds benefit (negative ROI).
        """
        if total_benefit_usd <= 0 or total_ai_cost_usd <= 0:
            return None
        daily_net_benefit = (total_benefit_usd - total_ai_cost_usd) / period_days
        if daily_net_benefit <= 0:
            return None
        return int(total_ai_cost_usd / daily_net_benefit)

    async def calculate_roi(
        self,
        tenant_id: str,
        initiative_name: str,
        initiative_type: str,
        period_start: datetime,
        period_end: datetime,
        hours_saved: float,
        headcount: int,
        hourly_rate_usd: float | None,
        error_reduction_rate: float,
        avg_error_cost_usd: float,
        incidents_prevented: int,
        avg_incident_cost_usd: float,
        additional_infra_cost_usd: float = 0.0,
        description: str | None = None,
    ) -> ROICalculation:
        """Calculate ROI for an AI initiative and persist the result.

        Automatically pulls GPU and token costs from the database for the
        measurement period to form the AI cost denominator.

        Args:
            tenant_id: The tenant running the AI initiative.
            initiative_name: Human-readable initiative name.
            initiative_type: Classification of the AI initiative.
            period_start: Start of the measurement window.
            period_end: End of the measurement window.
            hours_saved: Total hours of human labor saved.
            headcount: Number of team members benefiting.
            hourly_rate_usd: Fully-loaded hourly rate (defaults to settings value).
            error_reduction_rate: Fraction of errors eliminated (0.0–1.0).
            avg_error_cost_usd: Average cost per error before AI.
            incidents_prevented: Number of incidents prevented by AI.
            avg_incident_cost_usd: Average cost per incident.
            additional_infra_cost_usd: Any infrastructure costs not in cost_records.
            description: Optional description of measurement methodology.

        Returns:
            The persisted ROICalculation with computed ROI percent.
        """
        effective_hourly_rate = hourly_rate_usd or self._settings.default_hourly_rate_usd

        # Pull actual costs from the database for the period
        gpu_cost = await self._cost_repo.sum_cost_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            scope="gpu",
        )
        token_cost = await self._token_repo.sum_cost_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )
        infra_cost = await self._cost_repo.sum_cost_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            scope="all",
        ) - gpu_cost + additional_infra_cost_usd

        total_ai_cost_usd = gpu_cost + token_cost + infra_cost

        # Compute benefit components
        productivity_gain_usd = hours_saved * effective_hourly_rate * headcount
        quality_improvement_usd = error_reduction_rate * avg_error_cost_usd
        risk_avoidance_usd = incidents_prevented * avg_incident_cost_usd
        total_benefit_usd = productivity_gain_usd + quality_improvement_usd + risk_avoidance_usd

        # Compute ROI
        roi_percent = 0.0
        if total_ai_cost_usd > 0:
            roi_percent = (total_benefit_usd - total_ai_cost_usd) / total_ai_cost_usd * 100

        period_days = max(1, (period_end - period_start).days)
        payback_period_days = self._compute_payback_period(
            total_benefit_usd=total_benefit_usd,
            total_ai_cost_usd=total_ai_cost_usd,
            period_days=period_days,
        )

        assumptions: dict[str, Any] = {
            "hours_saved": hours_saved,
            "headcount": headcount,
            "hourly_rate_usd": effective_hourly_rate,
            "error_reduction_rate": error_reduction_rate,
            "avg_error_cost_usd": avg_error_cost_usd,
            "incidents_prevented": incidents_prevented,
            "avg_incident_cost_usd": avg_incident_cost_usd,
        }

        calculation = ROICalculation(
            tenant_id=tenant_id,
            initiative_name=initiative_name,
            initiative_type=initiative_type,
            description=description,
            period_start=period_start,
            period_end=period_end,
            productivity_gain_usd=productivity_gain_usd,
            quality_improvement_usd=quality_improvement_usd,
            risk_avoidance_usd=risk_avoidance_usd,
            total_benefit_usd=total_benefit_usd,
            gpu_cost_usd=gpu_cost,
            token_cost_usd=token_cost,
            infra_cost_usd=infra_cost,
            total_ai_cost_usd=total_ai_cost_usd,
            roi_percent=roi_percent,
            payback_period_days=payback_period_days,
            assumptions=assumptions,
            status="completed",
        )

        persisted = await self._roi_repo.create(calculation)

        await self._publisher.publish_roi_calculated(
            tenant_id=tenant_id,
            calculation_id=str(persisted.id),
            initiative_name=initiative_name,
            roi_percent=roi_percent,
            total_benefit_usd=total_benefit_usd,
            total_ai_cost_usd=total_ai_cost_usd,
        )

        logger.info(
            "roi_calculated",
            tenant_id=tenant_id,
            initiative_name=initiative_name,
            roi_percent=roi_percent,
            total_benefit_usd=total_benefit_usd,
            total_ai_cost_usd=total_ai_cost_usd,
        )

        return persisted

    async def list_reports(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ROICalculation]:
        """List ROI reports for a tenant."""
        return await self._roi_repo.list_by_tenant(
            tenant_id=tenant_id,
            status=status,
            limit=limit,
            offset=offset,
        )


class BudgetAlertService:
    """Evaluate spending against budget thresholds and trigger alerts.

    Runs on-demand (or scheduled externally) to compare actual spend
    with configured budgets. Creates BudgetAlert records and publishes
    finops.budget_exceeded Kafka events when thresholds are breached.
    """

    def __init__(
        self,
        budget_repo: IBudgetRepository,
        alert_repo: IBudgetAlertRepository,
        cost_repo: ICostRecordRepository,
        token_repo: ITokenUsageRepository,
        event_publisher: IFinOpsEventPublisher,
        settings: Settings,
    ) -> None:
        """Initialize BudgetAlertService with required dependencies."""
        self._budget_repo = budget_repo
        self._alert_repo = alert_repo
        self._cost_repo = cost_repo
        self._token_repo = token_repo
        self._publisher = event_publisher
        self._settings = settings

    async def create_budget(
        self,
        tenant_id: str,
        name: str,
        limit_usd: float,
        period_start: datetime,
        period_end: datetime,
        budget_type: str = "monthly",
        scope: str = "all",
        warning_threshold: float | None = None,
        critical_threshold: float | None = None,
        notification_channels: list[dict[str, Any]] | None = None,
    ) -> Budget:
        """Create a new budget threshold for a tenant.

        Args:
            tenant_id: The tenant for this budget.
            name: Human-readable budget name.
            limit_usd: Maximum spend allowed.
            period_start: Budget period start.
            period_end: Budget period end.
            budget_type: monthly | quarterly | annual | custom
            scope: all | gpu | tokens | model_id:{uuid}
            warning_threshold: Warning alert fraction (default from settings).
            critical_threshold: Critical alert fraction (default from settings).
            notification_channels: Alert destinations.

        Returns:
            The persisted Budget.
        """
        budget = Budget(
            tenant_id=tenant_id,
            name=name,
            limit_usd=limit_usd,
            period_start=period_start,
            period_end=period_end,
            budget_type=budget_type,
            scope=scope,
            warning_threshold=warning_threshold or self._settings.default_budget_alert_threshold,
            critical_threshold=critical_threshold or self._settings.critical_budget_alert_threshold,
            notification_channels=notification_channels or [],
            is_active=True,
        )

        persisted = await self._budget_repo.create(budget)

        logger.info(
            "budget_created",
            tenant_id=tenant_id,
            budget_id=str(persisted.id),
            name=name,
            limit_usd=limit_usd,
        )

        return persisted

    async def evaluate_budgets(
        self,
        tenant_id: str,
        at_time: datetime | None = None,
    ) -> list[BudgetAlert]:
        """Evaluate all active budgets for a tenant and create alerts as needed.

        Args:
            tenant_id: The tenant whose budgets to evaluate.
            at_time: Point in time to evaluate (defaults to now).

        Returns:
            List of newly created BudgetAlert records.
        """
        check_time = at_time or datetime.now(timezone.utc)
        active_budgets = await self._budget_repo.get_active_budgets_for_period(
            tenant_id=tenant_id,
            at_time=check_time,
        )

        new_alerts: list[BudgetAlert] = []

        for budget in active_budgets:
            actual_spend = await self._cost_repo.sum_cost_by_tenant_period(
                tenant_id=tenant_id,
                period_start=budget.period_start,
                period_end=check_time,
                scope=budget.scope,
            )
            # Add token costs if scope includes them
            if budget.scope in ("all", "tokens"):
                token_spend = await self._token_repo.sum_cost_by_tenant_period(
                    tenant_id=tenant_id,
                    period_start=budget.period_start,
                    period_end=check_time,
                )
                actual_spend += token_spend

            utilization = (actual_spend / budget.limit_usd) if budget.limit_usd > 0 else 0.0

            severity: str | None = None
            if utilization >= budget.critical_threshold:
                severity = "critical"
            elif utilization >= budget.warning_threshold:
                severity = "warning"

            if severity is not None:
                threshold_fraction = (
                    budget.critical_threshold if severity == "critical" else budget.warning_threshold
                )
                threshold_usd = budget.limit_usd * threshold_fraction

                alert = BudgetAlert(
                    tenant_id=tenant_id,
                    budget_id=budget.id,
                    severity=severity,
                    actual_spend_usd=actual_spend,
                    threshold_usd=threshold_usd,
                    utilization_percent=utilization * 100,
                    message=(
                        f"Budget '{budget.name}' has reached {utilization * 100:.1f}% utilization. "
                        f"Actual spend: ${actual_spend:,.2f} / ${budget.limit_usd:,.2f} limit."
                    ),
                    acknowledged=False,
                )

                persisted_alert = await self._alert_repo.create(alert)
                new_alerts.append(persisted_alert)

                await self._publisher.publish_budget_exceeded(
                    tenant_id=tenant_id,
                    budget_id=str(budget.id),
                    budget_name=budget.name,
                    severity=severity,
                    actual_spend_usd=actual_spend,
                    limit_usd=budget.limit_usd,
                    utilization_percent=utilization * 100,
                )

                logger.warning(
                    "budget_threshold_exceeded",
                    tenant_id=tenant_id,
                    budget_id=str(budget.id),
                    budget_name=budget.name,
                    severity=severity,
                    utilization_percent=utilization * 100,
                    actual_spend_usd=actual_spend,
                    limit_usd=budget.limit_usd,
                )

        return new_alerts

    async def get_budget_alerts(
        self,
        budget_id: uuid.UUID,
        acknowledged: bool | None = None,
    ) -> list[BudgetAlert]:
        """List alerts for a specific budget."""
        return await self._alert_repo.list_by_budget(
            budget_id=budget_id,
            acknowledged=acknowledged,
        )

    async def list_budgets(
        self,
        tenant_id: str,
        is_active: bool | None = None,
    ) -> list[Budget]:
        """List budgets for a tenant."""
        return await self._budget_repo.list_by_tenant(
            tenant_id=tenant_id,
            is_active=is_active,
        )


class RoutingOptimizerService:
    """Recommend cost-optimized model routing for AI workloads.

    Scores model candidates using a weighted multi-objective function:
        score = cost_weight × cost_score + quality_weight × quality_score
                + latency_weight × latency_score

    Higher score = better fit. Recommends the top-scoring model.
    """

    # Model catalog with cost tiers, quality ratings (0.0–1.0), and typical latency (ms)
    _MODEL_CATALOG: list[dict[str, Any]] = [
        {"name": "claude-opus-4", "tier": "tier1", "quality": 0.98, "typical_latency_ms": 3000, "provider": "anthropic"},
        {"name": "gpt-4o", "tier": "tier1", "quality": 0.96, "typical_latency_ms": 2500, "provider": "openai"},
        {"name": "claude-sonnet-4", "tier": "tier2", "quality": 0.90, "typical_latency_ms": 1500, "provider": "anthropic"},
        {"name": "gpt-4o-mini", "tier": "tier2", "quality": 0.85, "typical_latency_ms": 800, "provider": "openai"},
        {"name": "llama-3-70b", "tier": "tier2", "quality": 0.82, "typical_latency_ms": 1200, "provider": "self-hosted"},
        {"name": "claude-haiku-4", "tier": "tier3", "quality": 0.78, "typical_latency_ms": 400, "provider": "anthropic"},
        {"name": "gpt-3.5-turbo", "tier": "tier3", "quality": 0.72, "typical_latency_ms": 500, "provider": "openai"},
        {"name": "llama-3-8b", "tier": "tier3", "quality": 0.68, "typical_latency_ms": 300, "provider": "self-hosted"},
    ]

    # Minimum quality requirements per level
    _QUALITY_MINIMUMS: dict[str, float] = {
        "economy": 0.60,
        "standard": 0.78,
        "premium": 0.90,
    }

    def __init__(
        self,
        routing_repo: IRoutingRecommendationRepository,
        settings: Settings,
    ) -> None:
        """Initialize RoutingOptimizerService with required dependencies."""
        self._routing_repo = routing_repo
        self._settings = settings

    def _compute_token_cost_usd(
        self,
        tier: str,
        avg_prompt_tokens: int,
        avg_completion_tokens: int,
        monthly_requests: int,
    ) -> float:
        """Estimate monthly token cost for a model tier.

        Args:
            tier: tier1 | tier2 | tier3
            avg_prompt_tokens: Average prompt tokens per request.
            avg_completion_tokens: Average completion tokens per request.
            monthly_requests: Estimated monthly request volume.

        Returns:
            Estimated monthly cost in USD.
        """
        tier_costs: dict[str, float] = {
            "tier1": self._settings.token_cost_per_million_input_tier1,
            "tier2": self._settings.token_cost_per_million_input_tier2,
            "tier3": self._settings.token_cost_per_million_input_tier3,
        }
        input_cost = tier_costs.get(tier, self._settings.token_cost_per_million_input_tier2)
        output_cost = input_cost * 3.0

        prompt_cost = (avg_prompt_tokens / 1_000_000) * input_cost * monthly_requests
        completion_cost = (avg_completion_tokens / 1_000_000) * output_cost * monthly_requests
        return prompt_cost + completion_cost

    async def optimize_routing(
        self,
        tenant_id: str,
        workload_name: str,
        use_case: str,
        quality_requirement: str = "standard",
        latency_requirement_ms: int | None = None,
        estimated_monthly_requests: int = 1000,
        avg_prompt_tokens: int = 500,
        avg_completion_tokens: int = 200,
    ) -> RoutingRecommendation:
        """Generate a cost-optimized routing recommendation.

        Args:
            tenant_id: The tenant requesting routing guidance.
            workload_name: Service or agent name.
            use_case: Business use case for context.
            quality_requirement: economy | standard | premium
            latency_requirement_ms: Maximum acceptable latency (None = no constraint).
            estimated_monthly_requests: Volume for cost projection.
            avg_prompt_tokens: Average input tokens per call.
            avg_completion_tokens: Average output tokens per call.

        Returns:
            The persisted RoutingRecommendation.
        """
        min_quality = self._QUALITY_MINIMUMS.get(quality_requirement, 0.78)

        # Filter eligible candidates
        eligible = [
            m for m in self._MODEL_CATALOG
            if m["quality"] >= min_quality
            and (latency_requirement_ms is None or m["typical_latency_ms"] <= latency_requirement_ms)
        ]

        if not eligible:
            # Relax constraints — fall back to the best matching on quality only
            eligible = [m for m in self._MODEL_CATALOG if m["quality"] >= min_quality]

        # Score all candidates
        # Find max cost to normalize cost scores (lower cost = higher score)
        max_cost = max(
            self._compute_token_cost_usd(
                m["tier"], avg_prompt_tokens, avg_completion_tokens, estimated_monthly_requests
            )
            for m in eligible
        )

        scored_candidates: list[dict[str, Any]] = []
        for model in eligible:
            monthly_cost = self._compute_token_cost_usd(
                model["tier"], avg_prompt_tokens, avg_completion_tokens, estimated_monthly_requests
            )
            # Cost score: inverse of cost (0.0 = most expensive, 1.0 = cheapest)
            cost_score = 1.0 - (monthly_cost / max_cost) if max_cost > 0 else 1.0

            # Quality score: direct from catalog
            quality_score = model["quality"]

            # Latency score: lower latency = higher score
            max_latency = max(m["typical_latency_ms"] for m in eligible)
            latency_score = 1.0 - (model["typical_latency_ms"] / max_latency) if max_latency > 0 else 1.0

            composite_score = (
                self._settings.routing_cost_weight * cost_score
                + self._settings.routing_quality_weight * quality_score
                + self._settings.routing_latency_weight * latency_score
            )

            scored_candidates.append({
                "model_name": model["name"],
                "provider": model["provider"],
                "tier": model["tier"],
                "cost_score": round(cost_score, 4),
                "quality_score": round(quality_score, 4),
                "latency_score": round(latency_score, 4),
                "composite_score": round(composite_score, 4),
                "estimated_monthly_cost_usd": round(monthly_cost, 2),
                "typical_latency_ms": model["typical_latency_ms"],
            })

        # Sort by composite score descending
        scored_candidates.sort(key=lambda c: c["composite_score"], reverse=True)

        best = scored_candidates[0]
        fallback = scored_candidates[1] if len(scored_candidates) > 1 else None

        # Compute savings vs most expensive viable option
        max_viable_cost = max(c["estimated_monthly_cost_usd"] for c in scored_candidates)
        savings_usd = max_viable_cost - best["estimated_monthly_cost_usd"]

        reasoning = (
            f"Recommended {best['model_name']} (composite score: {best['composite_score']:.3f}). "
            f"Balances {quality_requirement} quality requirement with cost optimization. "
            f"Estimated monthly cost: ${best['estimated_monthly_cost_usd']:.2f}, "
            f"saving ${savings_usd:.2f}/month vs the most expensive viable option."
        )

        recommendation = RoutingRecommendation(
            tenant_id=tenant_id,
            workload_name=workload_name,
            use_case=use_case,
            quality_requirement=quality_requirement,
            latency_requirement_ms=latency_requirement_ms,
            estimated_monthly_requests=estimated_monthly_requests,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_completion_tokens=avg_completion_tokens,
            recommended_model_name=best["model_name"],
            fallback_model_name=fallback["model_name"] if fallback else None,
            routing_score=best["composite_score"],
            estimated_monthly_cost_usd=best["estimated_monthly_cost_usd"],
            estimated_savings_vs_premium_usd=savings_usd,
            reasoning=reasoning,
            candidate_models=scored_candidates,
        )

        persisted = await self._routing_repo.create(recommendation)

        logger.info(
            "routing_recommendation_generated",
            tenant_id=tenant_id,
            workload_name=workload_name,
            use_case=use_case,
            recommended_model=best["model_name"],
            routing_score=best["composite_score"],
            estimated_monthly_cost_usd=best["estimated_monthly_cost_usd"],
        )

        return persisted


class CostForecastService:
    """Provide cost trend analysis, projections, anomaly detection, and forecasting.

    Orchestrates ICostForecaster to produce insight reports from historical
    cost data without requiring direct repository access in the adapter layer.
    """

    def __init__(
        self,
        cost_repo: ICostRecordRepository,
        token_repo: ITokenUsageRepository,
        forecaster: ICostForecaster,
        settings: Settings,
    ) -> None:
        """Initialize CostForecastService with required dependencies.

        Args:
            cost_repo: Cost record repository for historical data retrieval.
            token_repo: Token usage repository for token cost data.
            forecaster: Cost forecasting adapter.
            settings: Service settings with thresholds and defaults.
        """
        self._cost_repo = cost_repo
        self._token_repo = token_repo
        self._forecaster = forecaster
        self._settings = settings

    async def generate_cost_forecast_report(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        projection_periods: int = 3,
        budget_usd: float | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive cost forecast report from historical data.

        Pulls cost and token records for the period, builds a cost history
        timeline, then delegates to ICostForecaster for analysis and projections.

        Args:
            tenant_id: Tenant for which to generate the report.
            period_start: Start of the historical window.
            period_end: End of the historical window.
            projection_periods: Number of forward periods to project.
            budget_usd: Optional total budget for exhaustion analysis.

        Returns:
            Complete forecast report dict with trends, projections, anomalies,
            and budget health summary.
        """
        cost_records = await self._cost_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )
        token_records = await self._token_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )

        # Build a unified cost history timeline (monthly buckets)
        cost_history = _build_cost_history(cost_records, token_records)

        report = await self._forecaster.generate_forecast_report(
            tenant_id=tenant_id,
            cost_history=cost_history,
            budget_usd=budget_usd,
            projection_periods=projection_periods,
        )

        logger.info(
            "cost_forecast_report_generated",
            tenant_id=tenant_id,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            projection_periods=projection_periods,
            history_data_points=len(cost_history),
        )

        return report

    async def detect_anomalies(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        z_score_threshold: float = 2.5,
    ) -> dict[str, Any]:
        """Detect anomalous cost spikes in the historical period.

        Args:
            tenant_id: Tenant for anomaly detection.
            period_start: Start of the historical window.
            period_end: End of the historical window.
            z_score_threshold: Z-score above which a period is flagged (default 2.5).

        Returns:
            Dict with anomalies list and anomaly_count.
        """
        cost_records = await self._cost_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )
        token_records = await self._token_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )
        cost_history = _build_cost_history(cost_records, token_records)

        result = await self._forecaster.detect_cost_anomalies(
            tenant_id=tenant_id,
            cost_history=cost_history,
            z_score_threshold=z_score_threshold,
        )

        logger.info(
            "cost_anomaly_detection_completed",
            tenant_id=tenant_id,
            anomaly_count=result.get("anomaly_count", 0),
            z_score_threshold=z_score_threshold,
        )

        return result


class InvoiceService:
    """Compile and manage per-tenant invoices and provider bill reconciliation.

    Orchestrates IInvoiceGenerator to produce structured invoice documents
    and reconciliation reports against provider billing data.
    """

    def __init__(
        self,
        cost_repo: ICostRecordRepository,
        token_repo: ITokenUsageRepository,
        invoice_generator: IInvoiceGenerator,
        event_publisher: IFinOpsEventPublisher,
        settings: Settings,
    ) -> None:
        """Initialize InvoiceService with required dependencies.

        Args:
            cost_repo: Cost record repository for billing period data.
            token_repo: Token usage repository for token billing data.
            invoice_generator: Invoice compilation adapter.
            event_publisher: FinOps event publisher.
            settings: Service settings.
        """
        self._cost_repo = cost_repo
        self._token_repo = token_repo
        self._invoice_generator = invoice_generator
        self._publisher = event_publisher
        self._settings = settings

    async def generate_invoice(
        self,
        tenant_id: str,
        billing_period_start: datetime,
        billing_period_end: datetime,
        tax_jurisdiction: str = "US",
        payment_terms_days: int = 30,
    ) -> dict[str, Any]:
        """Generate a complete invoice for a tenant billing period.

        Pulls all cost and token records for the period and delegates
        compilation to IInvoiceGenerator.

        Args:
            tenant_id: Tenant for which to generate the invoice.
            billing_period_start: Inclusive start of the billing period.
            billing_period_end: Inclusive end of the billing period.
            tax_jurisdiction: ISO country code for tax rate lookup (default US).
            payment_terms_days: Net payment terms in days (default 30).

        Returns:
            Complete invoice dict with line items, totals, and metadata.
        """
        cost_records = await self._cost_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=billing_period_start,
            period_end=billing_period_end,
        )
        token_records = await self._token_repo.list_by_tenant_period(
            tenant_id=tenant_id,
            period_start=billing_period_start,
            period_end=billing_period_end,
        )

        invoice = await self._invoice_generator.compile_tenant_invoice(
            tenant_id=tenant_id,
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            cost_records=cost_records,
            token_usage_records=token_records,
            tax_jurisdiction=tax_jurisdiction,
            payment_terms_days=payment_terms_days,
        )

        logger.info(
            "invoice_generated",
            tenant_id=tenant_id,
            invoice_number=invoice.get("invoice_number", ""),
            total_usd=invoice.get("total_usd", 0.0),
            billing_period_start=billing_period_start.isoformat(),
            billing_period_end=billing_period_end.isoformat(),
        )

        return invoice

    async def reconcile_with_provider(
        self,
        tenant_id: str,
        internal_invoice: dict[str, Any],
        provider_bill: dict[str, Any],
        tolerance_percent: float = 2.0,
    ) -> dict[str, Any]:
        """Reconcile an internal invoice against a provider bill.

        Args:
            tenant_id: Tenant for the reconciliation.
            internal_invoice: Invoice from generate_invoice.
            provider_bill: Provider bill dict with line_items and total.
            tolerance_percent: Acceptable variance percent (default 2.0%).

        Returns:
            Reconciliation report with matched items, variances, and status.
        """
        result = await self._invoice_generator.reconcile_with_provider_bill(
            tenant_id=tenant_id,
            internal_invoice=internal_invoice,
            provider_bill=provider_bill,
            tolerance_percent=tolerance_percent,
        )

        logger.info(
            "invoice_reconciliation_completed",
            tenant_id=tenant_id,
            reconciliation_status=result.get("reconciliation_status", "unknown"),
            variance_usd=result.get("variance_usd", 0.0),
            tolerance_percent=tolerance_percent,
        )

        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_cost_history(
    cost_records: list[Any],
    token_records: list[Any],
) -> list[dict[str, Any]]:
    """Aggregate cost and token records into monthly cost history buckets.

    Args:
        cost_records: List of CostRecord objects with period_start and cost_usd.
        token_records: List of TokenUsage objects with period_start and total_cost_usd.

    Returns:
        List of monthly cost dicts with period_label and total_cost_usd,
        sorted by period_label ascending.
    """
    from collections import defaultdict

    monthly_totals: dict[str, float] = defaultdict(float)

    for record in cost_records:
        period_start = getattr(record, "period_start", None)
        cost_usd = getattr(record, "cost_usd", 0.0) or 0.0
        if period_start is not None:
            period_key = period_start.strftime("%Y-%m")
            monthly_totals[period_key] += float(cost_usd)

    for record in token_records:
        period_start = getattr(record, "period_start", None)
        total_cost_usd = getattr(record, "total_cost_usd", 0.0) or 0.0
        if period_start is not None:
            period_key = period_start.strftime("%Y-%m")
            monthly_totals[period_key] += float(total_cost_usd)

    return [
        {"period_label": period, "total_cost_usd": round(total, 6)}
        for period, total in sorted(monthly_totals.items())
    ]
