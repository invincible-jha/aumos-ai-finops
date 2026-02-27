"""FastAPI routes for the cost-outcome attribution and ROI measurement API.

All routes are thin: validate inputs, call CostOutcomeAttributor, return
typed Pydantic response models. No business logic in this module.

Endpoints:
  GET  /api/v1/finops/roi/decision/{decision_id} — cost breakdown + outcome + ROI
  GET  /api/v1/finops/roi/use-case/{use_case}   — use case ROI summary
  GET  /api/v1/finops/roi/dashboard/{tenant_id} — full ROI dashboard
  POST /api/v1/finops/outcomes                  — submit business outcome event
  POST /api/v1/finops/outcome-adapters/configure — configure outcome adapter
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session

from aumos_ai_finops.api.schemas.attribution import (
    BreakEvenAnalysisResponse,
    DecisionCostBreakdown,
    DecisionROIResponse,
    OutcomeAdapterConfigRequest,
    OutcomeAdapterConfigResponse,
    ROIDashboardResponse,
    SubmitOutcomeRequest,
    SubmitOutcomeResponse,
    UseCaseROISummaryResponse,
)
from aumos_ai_finops.attribution.cost_outcome_attributor import (
    BreakEvenAnalysis,
    CostOutcomeAttributor,
    IAttributionRepository,
    OutcomeAttribution,
    UseCaseROISummary,
)
from aumos_ai_finops.attribution.outcome_adapters import IOutcomeAdapter
from aumos_ai_finops.attribution.outcome_adapters.manual import ManualEntryOutcomeAdapter
from aumos_ai_finops.attribution.outcome_adapters.webhook import WebhookOutcomeAdapter

router = APIRouter(prefix="/finops", tags=["finops-roi-attribution"])


# ---------------------------------------------------------------------------
# In-process singleton adapters (replace with app-state injection in prod)
# ---------------------------------------------------------------------------

_webhook_adapter = WebhookOutcomeAdapter()
_manual_adapter = ManualEntryOutcomeAdapter()


# ---------------------------------------------------------------------------
# Stub attribution repository
# ---------------------------------------------------------------------------


class _InMemoryAttributionRepository(IAttributionRepository):
    """In-memory stub repository for import/test verification.

    Production deployments inject a SQLAlchemy-backed implementation.
    """

    _store: dict[str, OutcomeAttribution] = {}

    async def save_attribution(self, attribution: OutcomeAttribution) -> None:
        self._store[f"{attribution.tenant_id}:{attribution.decision_id}"] = attribution

    async def get_by_decision_id(
        self,
        decision_id: str,
        tenant_id: str,
    ) -> OutcomeAttribution | None:
        return self._store.get(f"{tenant_id}:{decision_id}")

    async def list_pending_attributions(
        self,
        tenant_id: str,
        decision_after: datetime,
        limit: int = 100,
    ) -> list[OutcomeAttribution]:
        return [
            a
            for a in self._store.values()
            if a.tenant_id == tenant_id
            and not a.attribution_complete
            and a.decision_at >= decision_after
        ][:limit]

    async def get_use_case_summary(
        self,
        tenant_id: str,
        use_case: str,
        period_start: datetime,
        period_end: datetime,
    ) -> UseCaseROISummary | None:
        return None

    async def list_attributions_for_dashboard(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[OutcomeAttribution]:
        return [
            a
            for a in self._store.values()
            if a.tenant_id == tenant_id
            and period_start <= a.decision_at <= period_end
        ]


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def _get_attributor(
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> CostOutcomeAttributor:
    """Build CostOutcomeAttributor with stub dependencies.

    Args:
        session: SQLAlchemy async session (injected by FastAPI).

    Returns:
        CostOutcomeAttributor configured with in-memory stubs.
    """
    repo = _InMemoryAttributionRepository()
    return CostOutcomeAttributor(
        attribution_repo=repo,
        outcome_adapter=_manual_adapter,
    )


# ---------------------------------------------------------------------------
# Decision ROI endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/roi/decision/{decision_id}",
    response_model=DecisionROIResponse,
    summary="Get cost breakdown, outcome, and ROI for a single AI decision",
)
async def get_decision_roi(
    decision_id: str,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    attributor: Annotated[CostOutcomeAttributor, Depends(_get_attributor)] = ...,
) -> DecisionROIResponse:
    """Retrieve the full cost breakdown and ROI for a single AI decision.

    Returns all six cost components, the attributed business outcome (if any),
    and the computed ROI percentage.

    Args:
        decision_id: The UUID of the AI decision to retrieve.
        tenant: Authenticated tenant context.
        attributor: Injected CostOutcomeAttributor.

    Returns:
        DecisionROIResponse with cost breakdown and ROI.

    Raises:
        404: If no attribution record exists for this decision.
    """
    attribution = await attributor.get_decision_roi(
        decision_id=decision_id,
        tenant_id=tenant.tenant_id,
    )
    return _attribution_to_response(attribution)


# ---------------------------------------------------------------------------
# Use case ROI summary endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/roi/use-case/{use_case}",
    response_model=UseCaseROISummaryResponse,
    summary="Get aggregated ROI summary for a business use case",
)
async def get_use_case_roi_summary(
    use_case: str,
    period_start: Annotated[datetime, Query(description="Period start (UTC)")],
    period_end: Annotated[datetime, Query(description="Period end (UTC)")],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    attributor: Annotated[CostOutcomeAttributor, Depends(_get_attributor)] = ...,
) -> UseCaseROISummaryResponse:
    """Get aggregated cost and ROI metrics for a specific business use case.

    Aggregates all decisions within the period for the given use case and
    returns average ROI, total cost, total outcome value, and attribution rate.

    Args:
        use_case: Business use case label (URL-encoded if it contains spaces).
        period_start: Report period start (UTC).
        period_end: Report period end (UTC).
        tenant: Authenticated tenant context.
        attributor: Injected CostOutcomeAttributor.

    Returns:
        UseCaseROISummaryResponse with aggregated metrics.
    """
    summary = await attributor.get_use_case_roi_summary(
        tenant_id=tenant.tenant_id,
        use_case=use_case,
        period_start=period_start,
        period_end=period_end,
    )

    if summary is None:
        # No data: return empty summary
        return UseCaseROISummaryResponse(
            tenant_id=tenant.tenant_id,
            use_case=use_case,
            period_start=period_start,
            period_end=period_end,
            total_decisions=0,
            total_cost_usd=Decimal("0"),
            total_outcome_value_usd=Decimal("0"),
            avg_roi_pct=Decimal("0"),
            decisions_with_outcome=0,
            attribution_rate_pct=Decimal("0"),
        )

    attribution_rate = (
        Decimal(str(summary.decisions_with_outcome)) / Decimal(str(summary.total_decisions)) * Decimal("100")
        if summary.total_decisions > 0
        else Decimal("0")
    )

    return UseCaseROISummaryResponse(
        tenant_id=summary.tenant_id,
        use_case=summary.use_case,
        period_start=period_start,
        period_end=period_end,
        total_decisions=summary.total_decisions,
        total_cost_usd=summary.total_cost_usd,
        total_outcome_value_usd=summary.total_outcome_value_usd,
        avg_roi_pct=summary.avg_roi_pct,
        decisions_with_outcome=summary.decisions_with_outcome,
        attribution_rate_pct=attribution_rate.quantize(Decimal("0.01")),
    )


# ---------------------------------------------------------------------------
# ROI dashboard endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/roi/dashboard/{tenant_id}",
    response_model=ROIDashboardResponse,
    summary="Full ROI dashboard with per-use-case breakdown and break-even analysis",
)
async def get_roi_dashboard(
    tenant_id: str,
    period_start: Annotated[
        datetime,
        Query(description="Dashboard period start (UTC)"),
    ] = None,  # type: ignore[assignment]
    period_end: Annotated[
        datetime,
        Query(description="Dashboard period end (UTC)"),
    ] = None,  # type: ignore[assignment]
    period_days: Annotated[
        int,
        Query(ge=1, le=365, description="Look-back period in days (used when period_start/end not provided)"),
    ] = 30,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    attributor: Annotated[CostOutcomeAttributor, Depends(_get_attributor)] = ...,
) -> ROIDashboardResponse:
    """Get the full ROI dashboard for a tenant.

    Aggregates all attributions into per-use-case ROI summaries, performs
    break-even analysis for each use case, and computes overall portfolio ROI.

    Provide either period_start + period_end for a specific window, or use
    period_days for a trailing look-back from now.

    Args:
        tenant_id: Target tenant ID (must match authenticated tenant).
        period_start: Dashboard period start (UTC).
        period_end: Dashboard period end (UTC).
        period_days: Trailing look-back window in days (default 30).
        tenant: Authenticated tenant context.
        attributor: Injected CostOutcomeAttributor.

    Returns:
        ROIDashboardResponse with full portfolio and per-use-case breakdown.
    """
    now = datetime.now(timezone.utc)
    if period_end is None:
        period_end = now
    if period_start is None:
        from datetime import timedelta
        period_start = period_end - timedelta(days=period_days)

    dashboard_data = await attributor.get_roi_dashboard(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=period_end,
    )

    total_decisions = dashboard_data["total_decisions"]
    decisions_with_outcome = dashboard_data["decisions_with_outcome"]
    attribution_rate = (
        Decimal(str(decisions_with_outcome)) / Decimal(str(total_decisions)) * Decimal("100")
        if total_decisions > 0
        else Decimal("0")
    ).quantize(Decimal("0.01"))

    use_case_summaries = [
        _use_case_summary_to_response(s, period_start, period_end)
        for s in dashboard_data["use_case_summaries"]
    ]
    break_even_analyses = [
        _break_even_to_response(b)
        for b in dashboard_data["break_even_analyses"]
    ]

    return ROIDashboardResponse(
        tenant_id=tenant.tenant_id,
        period_start=period_start,
        period_end=period_end,
        total_decisions=total_decisions,
        total_cost_usd=dashboard_data["total_cost_usd"],
        total_outcome_value_usd=dashboard_data["total_outcome_value_usd"],
        portfolio_roi_pct=dashboard_data["portfolio_roi_pct"],
        decisions_with_outcome=decisions_with_outcome,
        attribution_rate_pct=attribution_rate,
        use_case_summaries=use_case_summaries,
        break_even_analyses=break_even_analyses,
    )


# ---------------------------------------------------------------------------
# Submit outcome endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/outcomes",
    response_model=SubmitOutcomeResponse,
    status_code=202,
    summary="Submit a business outcome event for attribution",
)
async def submit_outcome(
    request: SubmitOutcomeRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
    attributor: Annotated[CostOutcomeAttributor, Depends(_get_attributor)] = ...,
) -> SubmitOutcomeResponse:
    """Submit a business outcome event and attempt immediate attribution.

    If a cost record exists for the decision within the attribution window,
    ROI is computed immediately and the attribution is marked complete.
    Otherwise, the event is queued for processing when the cost record arrives.

    Args:
        request: Outcome submission payload.
        tenant: Authenticated tenant context.
        attributor: Injected CostOutcomeAttributor.

    Returns:
        SubmitOutcomeResponse with attribution status and ROI if available.
    """
    outcome_at = request.outcome_at or datetime.now(timezone.utc)

    if request.source == "webhook":
        await _webhook_adapter.enqueue_outcome(
            decision_id=request.decision_id,
            tenant_id=tenant.tenant_id,
            outcome_type=request.outcome_type,
            outcome_value_usd=request.outcome_value_usd,
            outcome_at=outcome_at,
        )
        return SubmitOutcomeResponse(
            decision_id=request.decision_id,
            outcome_type=request.outcome_type,
            outcome_value_usd=request.outcome_value_usd,
            attribution_complete=False,
            roi_pct=None,
            message="Outcome queued for processing via webhook adapter.",
        )

    # Manual source: attempt immediate attribution
    try:
        attribution = await attributor.attribute_outcome(
            decision_id=request.decision_id,
            tenant_id=tenant.tenant_id,
            outcome_type=request.outcome_type,
            outcome_value_usd=request.outcome_value_usd,
            outcome_at=outcome_at,
        )
        return SubmitOutcomeResponse(
            decision_id=request.decision_id,
            outcome_type=request.outcome_type,
            outcome_value_usd=request.outcome_value_usd,
            attribution_complete=True,
            roi_pct=attribution.roi_pct,
            message=f"Attribution complete. ROI: {attribution.roi_pct}%.",
        )
    except ValueError:
        # Decision not yet recorded — store for later processing
        await _manual_adapter.submit_outcome(
            decision_id=request.decision_id,
            tenant_id=tenant.tenant_id,
            outcome_type=request.outcome_type,
            outcome_value_usd=request.outcome_value_usd,
            outcome_at=outcome_at,
        )
        return SubmitOutcomeResponse(
            decision_id=request.decision_id,
            outcome_type=request.outcome_type,
            outcome_value_usd=request.outcome_value_usd,
            attribution_complete=False,
            roi_pct=None,
            message="Outcome stored. Will be attributed when the decision cost record is available.",
        )


# ---------------------------------------------------------------------------
# Configure outcome adapter endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/outcome-adapters/configure",
    response_model=OutcomeAdapterConfigResponse,
    status_code=200,
    summary="Configure an outcome adapter for a tenant",
)
async def configure_outcome_adapter(
    request: OutcomeAdapterConfigRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)] = ...,
) -> OutcomeAdapterConfigResponse:
    """Configure an outcome adapter for the current tenant.

    Sets up the adapter type and attribution window for outcome event sourcing.
    In production, webhook adapters are registered with the event bus to receive
    outcome payloads at the configured endpoint.

    Args:
        request: Adapter configuration payload.
        tenant: Authenticated tenant context.

    Returns:
        OutcomeAdapterConfigResponse confirming the configuration.
    """
    # In production: persist config to database and register webhook if needed.
    # Current implementation returns a confirmation with the requested settings.
    return OutcomeAdapterConfigResponse(
        tenant_id=tenant.tenant_id,
        adapter_type=request.adapter_type,
        attribution_window_days=request.attribution_window_days,
        enabled=request.enabled,
        configured_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Private serialization helpers
# ---------------------------------------------------------------------------


def _attribution_to_response(attribution: OutcomeAttribution) -> DecisionROIResponse:
    """Convert an OutcomeAttribution to a DecisionROIResponse.

    Args:
        attribution: The domain object to serialize.

    Returns:
        Pydantic response model.
    """
    costs = attribution.costs
    breakdown = DecisionCostBreakdown(
        input_token_cost=costs.input_token_cost,
        output_token_cost=costs.output_token_cost,
        compute_cost=costs.compute_cost,
        storage_cost=costs.storage_cost,
        egress_cost=costs.egress_cost,
        human_review_cost=costs.human_review_cost,
        total_cost_usd=costs.total_cost,
    )
    return DecisionROIResponse(
        decision_id=attribution.decision_id,
        tenant_id=attribution.tenant_id,
        ai_system_id=attribution.ai_system_id,
        use_case=attribution.use_case,
        cost_breakdown=breakdown,
        outcome_type=attribution.outcome_type,
        outcome_value_usd=attribution.outcome_value_usd,
        roi_pct=attribution.roi_pct,
        attribution_complete=attribution.attribution_complete,
        decision_at=attribution.decision_at,
        outcome_at=attribution.outcome_at,
    )


def _use_case_summary_to_response(
    summary: UseCaseROISummary,
    period_start: datetime,
    period_end: datetime,
) -> UseCaseROISummaryResponse:
    """Convert a UseCaseROISummary to the response schema.

    Args:
        summary: Domain summary object.
        period_start: Report period start.
        period_end: Report period end.

    Returns:
        UseCaseROISummaryResponse.
    """
    attribution_rate = (
        Decimal(str(summary.decisions_with_outcome)) / Decimal(str(summary.total_decisions)) * Decimal("100")
        if summary.total_decisions > 0
        else Decimal("0")
    ).quantize(Decimal("0.01"))

    return UseCaseROISummaryResponse(
        tenant_id=summary.tenant_id,
        use_case=summary.use_case,
        period_start=period_start,
        period_end=period_end,
        total_decisions=summary.total_decisions,
        total_cost_usd=summary.total_cost_usd,
        total_outcome_value_usd=summary.total_outcome_value_usd,
        avg_roi_pct=summary.avg_roi_pct,
        decisions_with_outcome=summary.decisions_with_outcome,
        attribution_rate_pct=attribution_rate,
    )


def _break_even_to_response(bea: BreakEvenAnalysis) -> BreakEvenAnalysisResponse:
    """Convert a BreakEvenAnalysis to the response schema.

    Args:
        bea: Domain break-even analysis object.

    Returns:
        BreakEvenAnalysisResponse.
    """
    return BreakEvenAnalysisResponse(
        use_case=bea.use_case,
        total_cost_usd=bea.total_cost_usd,
        total_outcome_value_usd=bea.total_outcome_value_usd,
        net_value_usd=bea.net_value_usd,
        roi_pct=bea.roi_pct,
        is_break_even=bea.is_break_even,
        days_to_break_even=bea.days_to_break_even,
    )
