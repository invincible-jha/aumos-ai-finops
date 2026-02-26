"""FastAPI router for the AumOS AI FinOps API.

All routes are thin — they validate inputs, call services, and return
Pydantic response models. No business logic belongs here.

Endpoints:
  GET    /api/v1/finops/costs                    Cost breakdown by resource/tenant
  GET    /api/v1/finops/costs/gpu                GPU-specific cost tracking
  POST   /api/v1/finops/costs/sync/opencost      Sync costs from OpenCost
  POST   /api/v1/finops/costs/record             Manually record a cost entry
  GET    /api/v1/finops/tokens                   Token consumption analytics
  GET    /api/v1/finops/tokens/by-model          Per-model token usage aggregation
  POST   /api/v1/finops/tokens/record            Record token usage
  POST   /api/v1/finops/roi/calculate            Calculate ROI for an AI initiative
  GET    /api/v1/finops/roi/reports              List ROI reports
  POST   /api/v1/finops/budgets                  Create budget threshold
  GET    /api/v1/finops/budgets                  List budgets
  GET    /api/v1/finops/budgets/{id}/alerts      Budget alerts for a specific budget
  POST   /api/v1/finops/budgets/evaluate         Evaluate all budgets and trigger alerts
  GET    /api/v1/finops/dashboard                Executive dashboard data
  POST   /api/v1/finops/routing/optimize         Cost-optimized model routing recommendation
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session
from aumos_common.errors import NotFoundError, ErrorCode

from aumos_ai_finops.adapters.kafka import FinOpsEventPublisher
from aumos_ai_finops.adapters.kubecost_client import KubeCostClient
from aumos_ai_finops.adapters.opencost_client import OpenCostClient
from aumos_ai_finops.adapters.repositories import (
    BudgetAlertRepository,
    BudgetRepository,
    CostRecordRepository,
    ROICalculationRepository,
    RoutingRecommendationRepository,
    TokenUsageRepository,
)
from aumos_ai_finops.api.schemas import (
    BudgetAlertResponse,
    BudgetResponse,
    CostRecordResponse,
    CreateBudgetRequest,
    DashboardSummaryResponse,
    ModelTokenAggregateResponse,
    ROICalculateRequest,
    ROICalculationResponse,
    RecordCostRequest,
    RecordTokenUsageRequest,
    RoutingOptimizeRequest,
    RoutingRecommendationResponse,
    SyncOpenCostRequest,
    TokenUsageResponse,
)
from aumos_ai_finops.core.services import (
    BudgetAlertService,
    CostCollectorService,
    ROIEngineService,
    RoutingOptimizerService,
    TokenTrackerService,
)
from aumos_ai_finops.settings import Settings

router = APIRouter(prefix="/finops", tags=["finops"])
settings = Settings()


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def _get_cost_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CostCollectorService:
    """Build CostCollectorService with all required dependencies."""
    cost_repo = CostRecordRepository(session)
    opencost_client = OpenCostClient(settings) if settings.opencost_enabled else None
    kubecost_client = KubeCostClient(settings) if settings.kubecost_enabled else None
    # EventPublisher initialized with a no-op placeholder for now
    # In production, inject from app lifespan via a global publisher
    from aumos_common.events import EventPublisher
    publisher = FinOpsEventPublisher(EventPublisher(settings=settings))
    return CostCollectorService(
        cost_repo=cost_repo,
        opencost_client=opencost_client,
        kubecost_client=kubecost_client,
        event_publisher=publisher,
        settings=settings,
    )


def _get_token_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TokenTrackerService:
    """Build TokenTrackerService with all required dependencies."""
    token_repo = TokenUsageRepository(session)
    return TokenTrackerService(token_repo=token_repo, settings=settings)


def _get_roi_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ROIEngineService:
    """Build ROIEngineService with all required dependencies."""
    from aumos_common.events import EventPublisher
    roi_repo = ROICalculationRepository(session)
    cost_repo = CostRecordRepository(session)
    token_repo = TokenUsageRepository(session)
    publisher = FinOpsEventPublisher(EventPublisher(settings=settings))
    return ROIEngineService(
        roi_repo=roi_repo,
        cost_repo=cost_repo,
        token_repo=token_repo,
        event_publisher=publisher,
        settings=settings,
    )


def _get_budget_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> BudgetAlertService:
    """Build BudgetAlertService with all required dependencies."""
    from aumos_common.events import EventPublisher
    budget_repo = BudgetRepository(session)
    alert_repo = BudgetAlertRepository(session)
    cost_repo = CostRecordRepository(session)
    token_repo = TokenUsageRepository(session)
    publisher = FinOpsEventPublisher(EventPublisher(settings=settings))
    return BudgetAlertService(
        budget_repo=budget_repo,
        alert_repo=alert_repo,
        cost_repo=cost_repo,
        token_repo=token_repo,
        event_publisher=publisher,
        settings=settings,
    )


def _get_routing_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> RoutingOptimizerService:
    """Build RoutingOptimizerService with all required dependencies."""
    routing_repo = RoutingRecommendationRepository(session)
    return RoutingOptimizerService(routing_repo=routing_repo, settings=settings)


# ---------------------------------------------------------------------------
# Cost endpoints
# ---------------------------------------------------------------------------


@router.get("/costs", response_model=list[CostRecordResponse], summary="Get cost breakdown")
async def get_costs(
    period_start: Annotated[datetime, Query(description="Start of the query window (UTC)")],
    period_end: Annotated[datetime, Query(description="End of the query window (UTC)")],
    resource_type: Annotated[str | None, Query(description="Filter by resource type")] = None,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[CostCollectorService, Depends(_get_cost_service)] = ...,
) -> list[CostRecordResponse]:
    """Get cost breakdown for the current tenant within a time window.

    Optionally filter by resource type (gpu | cpu | storage | network | memory).
    Results are ordered by period_start descending.
    """
    records = await service.get_cost_breakdown(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=period_end,
        resource_type=resource_type,
    )
    return [CostRecordResponse.model_validate(r) for r in records]


@router.get("/costs/gpu", response_model=list[CostRecordResponse], summary="Get GPU cost tracking")
async def get_gpu_costs(
    period_start: Annotated[datetime, Query()],
    period_end: Annotated[datetime, Query()],
    gpu_type: Annotated[str | None, Query(description="Filter by GPU model: a100 | h100 | t4")] = None,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[CostCollectorService, Depends(_get_cost_service)] = ...,
) -> list[CostRecordResponse]:
    """Get GPU-specific cost records for the current tenant.

    Optionally filter by GPU model (a100 | h100 | t4 | v100 | a10 | l4).
    """
    records = await service.get_gpu_costs(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=period_end,
        gpu_type=gpu_type,
    )
    return [CostRecordResponse.model_validate(r) for r in records]


@router.post(
    "/costs/record",
    response_model=CostRecordResponse,
    status_code=201,
    summary="Manually record a cost entry",
)
async def record_cost(
    request: RecordCostRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[CostCollectorService, Depends(_get_cost_service)] = ...,
) -> CostRecordResponse:
    """Manually record a cost allocation entry.

    Use this endpoint to record costs from providers not covered by
    the OpenCost/KubeCost integrations.
    """
    record = await service.record_cost(
        tenant_id=tenant.tenant_id,
        resource_type=request.resource_type,
        resource_id=request.resource_id,
        cost_usd=request.cost_usd,
        period_start=request.period_start,
        period_end=request.period_end,
        gpu_type=request.gpu_type,
        workload_name=request.workload_name,
        namespace=request.namespace,
        model_id=request.model_id,
        on_demand_cost_usd=request.on_demand_cost_usd,
        efficiency_rate=request.efficiency_rate,
        source=request.source,
    )
    return CostRecordResponse.model_validate(record)


@router.post(
    "/costs/sync/opencost",
    response_model=list[CostRecordResponse],
    summary="Sync costs from OpenCost",
)
async def sync_from_opencost(
    request: SyncOpenCostRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[CostCollectorService, Depends(_get_cost_service)] = ...,
) -> list[CostRecordResponse]:
    """Pull cost allocation data from OpenCost and persist as cost records.

    Requires AUMOS_FINOPS_OPENCOST_ENABLED=true in service configuration.
    """
    records = await service.sync_from_opencost(
        tenant_id=tenant.tenant_id,
        namespace=request.namespace,
        window=request.window,
    )
    return [CostRecordResponse.model_validate(r) for r in records]


# ---------------------------------------------------------------------------
# Token endpoints
# ---------------------------------------------------------------------------


@router.get("/tokens", response_model=list[TokenUsageResponse], summary="Get token consumption analytics")
async def get_tokens(
    period_start: Annotated[datetime, Query()],
    period_end: Annotated[datetime, Query()],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[TokenTrackerService, Depends(_get_token_service)] = ...,
) -> list[TokenUsageResponse]:
    """Get token consumption records for the current tenant in a time window."""
    usages = await service.get_token_analytics(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=period_end,
    )
    return [TokenUsageResponse.model_validate(u) for u in usages]


@router.get(
    "/tokens/by-model",
    response_model=list[ModelTokenAggregateResponse],
    summary="Get per-model token usage",
)
async def get_tokens_by_model(
    period_start: Annotated[datetime, Query()],
    period_end: Annotated[datetime, Query()],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[TokenTrackerService, Depends(_get_token_service)] = ...,
) -> list[ModelTokenAggregateResponse]:
    """Get token usage aggregated by model for the current tenant.

    Returns models sorted by total cost descending — useful for identifying
    expensive models and optimization opportunities.
    """
    aggregates = await service.get_usage_by_model(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=period_end,
    )
    return [ModelTokenAggregateResponse(**agg) for agg in aggregates]


@router.post(
    "/tokens/record",
    response_model=TokenUsageResponse,
    status_code=201,
    summary="Record token usage",
)
async def record_token_usage(
    request: RecordTokenUsageRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[TokenTrackerService, Depends(_get_token_service)] = ...,
) -> TokenUsageResponse:
    """Record LLM token consumption for the current tenant.

    Token costs are automatically calculated based on the model tier.
    """
    usage = await service.record_token_usage(
        tenant_id=tenant.tenant_id,
        model_id=request.model_id,
        model_name=request.model_name,
        model_provider=request.model_provider,
        model_tier=request.model_tier,
        period_start=request.period_start,
        period_end=request.period_end,
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
        request_count=request.request_count,
        workload_name=request.workload_name,
        use_case=request.use_case,
    )
    return TokenUsageResponse.model_validate(usage)


# ---------------------------------------------------------------------------
# ROI endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/roi/calculate",
    response_model=ROICalculationResponse,
    status_code=201,
    summary="Calculate ROI for an AI initiative",
)
async def calculate_roi(
    request: ROICalculateRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[ROIEngineService, Depends(_get_roi_service)] = ...,
) -> ROICalculationResponse:
    """Calculate multi-touch attribution ROI for an AI initiative.

    Automatically pulls GPU and token costs from the database for the
    specified period. Returns ROI percent, payback period, and full
    benefit/cost breakdown.
    """
    calculation = await service.calculate_roi(
        tenant_id=tenant.tenant_id,
        initiative_name=request.initiative_name,
        initiative_type=request.initiative_type,
        period_start=request.period_start,
        period_end=request.period_end,
        hours_saved=request.hours_saved,
        headcount=request.headcount,
        hourly_rate_usd=request.hourly_rate_usd,
        error_reduction_rate=request.error_reduction_rate,
        avg_error_cost_usd=request.avg_error_cost_usd,
        incidents_prevented=request.incidents_prevented,
        avg_incident_cost_usd=request.avg_incident_cost_usd,
        additional_infra_cost_usd=request.additional_infra_cost_usd,
        description=request.description,
    )
    return ROICalculationResponse.model_validate(calculation)


@router.get(
    "/roi/reports",
    response_model=list[ROICalculationResponse],
    summary="List ROI reports",
)
async def list_roi_reports(
    status: Annotated[str | None, Query(description="Filter by status: completed | draft | archived")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[ROIEngineService, Depends(_get_roi_service)] = ...,
) -> list[ROICalculationResponse]:
    """List ROI calculation reports for the current tenant.

    Results are ordered by creation date descending (most recent first).
    """
    reports = await service.list_reports(
        tenant_id=tenant.tenant_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return [ROICalculationResponse.model_validate(r) for r in reports]


# ---------------------------------------------------------------------------
# Budget endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/budgets",
    response_model=BudgetResponse,
    status_code=201,
    summary="Create a budget threshold",
)
async def create_budget(
    request: CreateBudgetRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[BudgetAlertService, Depends(_get_budget_service)] = ...,
) -> BudgetResponse:
    """Create a new budget threshold for the current tenant.

    Budgets trigger warning and critical alerts when actual spend
    crosses the configured threshold percentages.
    """
    budget = await service.create_budget(
        tenant_id=tenant.tenant_id,
        name=request.name,
        limit_usd=request.limit_usd,
        period_start=request.period_start,
        period_end=request.period_end,
        budget_type=request.budget_type,
        scope=request.scope,
        warning_threshold=request.warning_threshold,
        critical_threshold=request.critical_threshold,
        notification_channels=request.notification_channels,
    )
    return BudgetResponse.model_validate(budget)


@router.get(
    "/budgets",
    response_model=list[BudgetResponse],
    summary="List budgets",
)
async def list_budgets(
    is_active: Annotated[bool | None, Query(description="Filter by active status")] = None,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[BudgetAlertService, Depends(_get_budget_service)] = ...,
) -> list[BudgetResponse]:
    """List all budgets for the current tenant."""
    budgets = await service.list_budgets(
        tenant_id=tenant.tenant_id,
        is_active=is_active,
    )
    return [BudgetResponse.model_validate(b) for b in budgets]


@router.get(
    "/budgets/{budget_id}/alerts",
    response_model=list[BudgetAlertResponse],
    summary="Get budget alerts",
)
async def get_budget_alerts(
    budget_id: uuid.UUID,
    acknowledged: Annotated[bool | None, Query(description="Filter by acknowledgement status")] = None,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[BudgetAlertService, Depends(_get_budget_service)] = ...,
) -> list[BudgetAlertResponse]:
    """Get alerts for a specific budget.

    Returns all triggered alerts, optionally filtered by acknowledgement status.
    """
    alerts = await service.get_budget_alerts(
        budget_id=budget_id,
        acknowledged=acknowledged,
    )
    return [BudgetAlertResponse.model_validate(a) for a in alerts]


@router.post(
    "/budgets/evaluate",
    response_model=list[BudgetAlertResponse],
    summary="Evaluate budgets and trigger alerts",
)
async def evaluate_budgets(
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[BudgetAlertService, Depends(_get_budget_service)] = ...,
) -> list[BudgetAlertResponse]:
    """Evaluate all active budgets for the current tenant and trigger alerts.

    Compares actual spend against all active budgets and creates
    BudgetAlert records (plus Kafka events) for any breached thresholds.
    Typically called by a scheduler, but can be triggered on-demand.
    """
    alerts = await service.evaluate_budgets(tenant_id=tenant.tenant_id)
    return [BudgetAlertResponse.model_validate(a) for a in alerts]


# ---------------------------------------------------------------------------
# Dashboard endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/dashboard",
    response_model=DashboardSummaryResponse,
    summary="Get executive dashboard data",
)
async def get_dashboard(
    period_days: Annotated[int, Query(ge=1, le=365, description="Look-back period in days")] = 30,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    cost_service: Annotated[CostCollectorService, Depends(_get_cost_service)] = ...,
    token_service: Annotated[TokenTrackerService, Depends(_get_token_service)] = ...,
    roi_service: Annotated[ROIEngineService, Depends(_get_roi_service)] = ...,
    budget_service: Annotated[BudgetAlertService, Depends(_get_budget_service)] = ...,
    session: Annotated[AsyncSession, Depends(get_db_session)] = ...,
) -> DashboardSummaryResponse:
    """Aggregate cost, token, and ROI metrics for the executive dashboard.

    Returns a single summary view covering the specified look-back period.
    """
    now = datetime.now(timezone.utc)
    period_start = now - timedelta(days=period_days)
    prior_period_start = period_start - timedelta(days=period_days)

    # Gather current period data
    cost_records = await cost_service.get_cost_breakdown(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=now,
    )
    gpu_records = await cost_service.get_gpu_costs(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=now,
    )
    token_aggregates = await token_service.get_usage_by_model(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=now,
    )

    total_cost_usd = sum(r.cost_usd for r in cost_records)
    gpu_cost_usd = sum(r.cost_usd for r in gpu_records)
    token_cost_usd = sum(a["total_cost_usd"] for a in token_aggregates)
    infra_cost_usd = total_cost_usd - gpu_cost_usd
    total_tokens = sum(a["total_tokens"] for a in token_aggregates)
    total_requests = sum(a["request_count"] for a in token_aggregates)

    # ROI — get latest completed calculation
    roi_reports = await roi_service.list_reports(
        tenant_id=tenant.tenant_id,
        status="completed",
        limit=1,
    )
    latest_roi_percent = roi_reports[0].roi_percent if roi_reports else None

    # Budget summary
    budgets = await budget_service.list_budgets(
        tenant_id=tenant.tenant_id,
        is_active=True,
    )
    active_budgets = len(budgets)
    budget_utilization_percent = None
    if budgets:
        # Simple average utilization across active budgets
        budget_limit_total = sum(b.limit_usd for b in budgets)
        if budget_limit_total > 0:
            budget_utilization_percent = (total_cost_usd + token_cost_usd) / budget_limit_total * 100

    # Unacknowledged alerts across all budgets
    alert_repo = BudgetAlertRepository(session)
    unacknowledged_alerts = 0
    for budget in budgets:
        unacked = await alert_repo.list_by_budget(budget_id=budget.id, acknowledged=False)
        unacknowledged_alerts += len(unacked)

    # Cost trend vs prior period
    cost_trend_percent = None
    prior_records = await cost_service.get_cost_breakdown(
        tenant_id=tenant.tenant_id,
        period_start=prior_period_start,
        period_end=period_start,
    )
    prior_total = sum(r.cost_usd for r in prior_records)
    if prior_total > 0:
        cost_trend_percent = (total_cost_usd - prior_total) / prior_total * 100

    top_models = [
        ModelTokenAggregateResponse(**agg)
        for agg in sorted(token_aggregates, key=lambda a: a["total_cost_usd"], reverse=True)[:5]
    ]

    return DashboardSummaryResponse(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=now,
        total_cost_usd=total_cost_usd + token_cost_usd,
        gpu_cost_usd=gpu_cost_usd,
        token_cost_usd=token_cost_usd,
        infra_cost_usd=infra_cost_usd,
        total_tokens=total_tokens,
        total_requests=total_requests,
        active_budgets=active_budgets,
        budget_utilization_percent=budget_utilization_percent,
        latest_roi_percent=latest_roi_percent,
        cost_trend_percent=cost_trend_percent,
        top_models_by_cost=top_models,
        unacknowledged_alerts=unacknowledged_alerts,
    )


# ---------------------------------------------------------------------------
# Routing endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/routing/optimize",
    response_model=RoutingRecommendationResponse,
    status_code=201,
    summary="Get cost-optimized model routing recommendation",
)
async def optimize_routing(
    request: RoutingOptimizeRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    service: Annotated[RoutingOptimizerService, Depends(_get_routing_service)] = ...,
) -> RoutingRecommendationResponse:
    """Generate a cost-optimized model routing recommendation.

    Evaluates available models against cost, quality, and latency objectives
    using a weighted scoring function. Returns the optimal model recommendation
    with projected monthly costs and savings vs the most expensive viable option.
    """
    recommendation = await service.optimize_routing(
        tenant_id=tenant.tenant_id,
        workload_name=request.workload_name,
        use_case=request.use_case,
        quality_requirement=request.quality_requirement,
        latency_requirement_ms=request.latency_requirement_ms,
        estimated_monthly_requests=request.estimated_monthly_requests,
        avg_prompt_tokens=request.avg_prompt_tokens,
        avg_completion_tokens=request.avg_completion_tokens,
    )
    return RoutingRecommendationResponse.model_validate(recommendation)
