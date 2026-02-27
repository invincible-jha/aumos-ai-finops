"""Pydantic request/response schemas for the cost-outcome attribution API.

All monetary fields use Decimal for precision. Datetime fields are UTC.

Endpoints served:
  GET  /api/v1/finops/roi/decision/{decision_id}
  GET  /api/v1/finops/roi/use-case/{use_case}
  GET  /api/v1/finops/roi/dashboard/{tenant_id}
  POST /api/v1/finops/outcomes
  POST /api/v1/finops/outcome-adapters/configure
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Cost component breakdown schema
# ---------------------------------------------------------------------------


class DecisionCostBreakdown(BaseModel):
    """Six-component cost breakdown for a single AI decision.

    Attributes:
        input_token_cost: Cost of prompt/input tokens.
        output_token_cost: Cost of completion/output tokens.
        compute_cost: GPU/CPU inference compute cost.
        storage_cost: Storage cost attributed to this decision.
        egress_cost: Network egress cost for the response.
        human_review_cost: Human review cost (0 if no review required).
        total_cost_usd: Sum of all six components.
    """

    input_token_cost: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Cost of prompt/input tokens in USD",
    )
    output_token_cost: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Cost of completion/output tokens in USD",
    )
    compute_cost: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="GPU/CPU inference compute cost in USD",
    )
    storage_cost: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Storage cost attributed to this decision in USD",
    )
    egress_cost: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Network egress cost for response delivery in USD",
    )
    human_review_cost: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Human review cost in USD (0 if no review)",
    )
    total_cost_usd: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Sum of all six cost components in USD",
    )


# ---------------------------------------------------------------------------
# Decision ROI response
# ---------------------------------------------------------------------------


class DecisionROIResponse(BaseModel):
    """Full cost breakdown and ROI result for a single AI decision.

    Attributes:
        decision_id: The AI decision identifier.
        tenant_id: Owning tenant.
        ai_system_id: AI system that made the decision.
        use_case: Business use case label.
        cost_breakdown: Decomposed cost components.
        outcome_type: Category of attributed outcome (None if not yet attributed).
        outcome_value_usd: Monetized outcome value (None if pending).
        roi_pct: ROI percentage (None if not yet attributed).
        attribution_complete: True when an outcome has been matched.
        decision_at: When the AI decision was made.
        outcome_at: When the outcome was realized (None if pending).
    """

    decision_id: str
    tenant_id: str
    ai_system_id: str
    use_case: str
    cost_breakdown: DecisionCostBreakdown
    outcome_type: str | None
    outcome_value_usd: Decimal | None
    roi_pct: Decimal | None
    attribution_complete: bool
    decision_at: datetime
    outcome_at: datetime | None


# ---------------------------------------------------------------------------
# Use case ROI summary response
# ---------------------------------------------------------------------------


class UseCaseROISummaryResponse(BaseModel):
    """Aggregated ROI summary for a business use case over a period.

    Attributes:
        tenant_id: Owning tenant.
        use_case: Business use case label.
        period_start: Summary period start.
        period_end: Summary period end.
        total_decisions: Total decisions processed.
        total_cost_usd: Sum of all decision costs.
        total_outcome_value_usd: Sum of all attributed outcome values.
        avg_roi_pct: Average ROI across decisions with outcomes.
        decisions_with_outcome: Count of decisions with an attributed outcome.
        attribution_rate_pct: decisions_with_outcome / total_decisions * 100.
    """

    tenant_id: str
    use_case: str
    period_start: datetime
    period_end: datetime
    total_decisions: int
    total_cost_usd: Decimal
    total_outcome_value_usd: Decimal
    avg_roi_pct: Decimal
    decisions_with_outcome: int
    attribution_rate_pct: Decimal


# ---------------------------------------------------------------------------
# Break-even analysis schema
# ---------------------------------------------------------------------------


class BreakEvenAnalysisResponse(BaseModel):
    """Break-even analysis result for a use case.

    Attributes:
        use_case: Use case label.
        total_cost_usd: Total investment cost.
        total_outcome_value_usd: Total realized outcome value.
        net_value_usd: Outcome minus cost (positive = profit).
        roi_pct: Return on investment percentage.
        is_break_even: True when outcomes exceed costs.
        days_to_break_even: Projected days to break even (None if unknown).
    """

    use_case: str
    total_cost_usd: Decimal
    total_outcome_value_usd: Decimal
    net_value_usd: Decimal
    roi_pct: Decimal
    is_break_even: bool
    days_to_break_even: int | None


# ---------------------------------------------------------------------------
# ROI dashboard response
# ---------------------------------------------------------------------------


class ROIDashboardResponse(BaseModel):
    """Full ROI dashboard for a tenant over a reporting period.

    Attributes:
        tenant_id: The tenant this dashboard covers.
        period_start: Dashboard period start (UTC).
        period_end: Dashboard period end (UTC).
        total_decisions: Total AI decisions in the period.
        total_cost_usd: Total cost across all decisions.
        total_outcome_value_usd: Total attributed outcome value.
        portfolio_roi_pct: Overall portfolio ROI percentage.
        decisions_with_outcome: Decisions with a matched outcome.
        attribution_rate_pct: Percentage of decisions with outcomes.
        use_case_summaries: Per use-case ROI summaries.
        break_even_analyses: Break-even analysis per use case.
    """

    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_decisions: int
    total_cost_usd: Decimal
    total_outcome_value_usd: Decimal
    portfolio_roi_pct: Decimal
    decisions_with_outcome: int
    attribution_rate_pct: Decimal
    use_case_summaries: list[UseCaseROISummaryResponse]
    break_even_analyses: list[BreakEvenAnalysisResponse]


# ---------------------------------------------------------------------------
# Submit outcome request
# ---------------------------------------------------------------------------


class SubmitOutcomeRequest(BaseModel):
    """Request body for submitting a business outcome event.

    Attributes:
        decision_id: UUID of the AI decision being attributed.
        outcome_type: Categorical type of the business outcome.
        outcome_value_usd: USD value of the realized outcome.
        outcome_at: When the outcome was realized (defaults to now UTC if omitted).
        source: Outcome source: webhook | manual.
        raw_value: Optional raw metric value before monetization.
        raw_unit: Unit for raw_value (e.g., hours, incidents, conversions).
        ai_system_id: AI system that produced the decision.
        use_case: Business use case label.
    """

    decision_id: str = Field(
        ...,
        description="UUID of the AI decision being attributed",
    )
    outcome_type: Literal[
        "revenue_generated",
        "cost_saved",
        "risk_avoided",
        "error_prevented",
        "time_saved",
        "conversion",
        "churn_prevented",
        "fraud_blocked",
        "compliance_upheld",
        "custom",
    ] = Field(
        ...,
        description="Categorical type of the business outcome",
    )
    outcome_value_usd: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Monetized USD value of the realized outcome",
    )
    outcome_at: datetime | None = Field(
        default=None,
        description="When the outcome was realized (UTC). Defaults to server time if omitted.",
    )
    source: Literal["webhook", "manual"] = Field(
        default="manual",
        description="Source of this outcome event",
    )
    raw_value: float | None = Field(
        default=None,
        description="Raw numeric metric before monetization (e.g., 2.5 for 2.5 hours saved)",
    )
    raw_unit: str | None = Field(
        default=None,
        max_length=100,
        description="Unit for raw_value (e.g., hours, incidents, conversions)",
    )
    ai_system_id: str = Field(
        ...,
        max_length=255,
        description="AI system that produced the decision",
    )
    use_case: str = Field(
        ...,
        max_length=255,
        description="Business use case label",
    )


class SubmitOutcomeResponse(BaseModel):
    """Response after submitting a business outcome event.

    Attributes:
        decision_id: The attributed decision.
        outcome_type: The outcome type submitted.
        outcome_value_usd: The outcome value recorded.
        attribution_complete: Whether attribution was completed immediately.
        roi_pct: ROI if attribution completed (None otherwise).
        message: Human-readable status message.
    """

    decision_id: str
    outcome_type: str
    outcome_value_usd: Decimal
    attribution_complete: bool
    roi_pct: Decimal | None
    message: str


# ---------------------------------------------------------------------------
# Outcome adapter configuration
# ---------------------------------------------------------------------------


class OutcomeAdapterConfigRequest(BaseModel):
    """Request body to configure an outcome adapter for a tenant.

    Attributes:
        adapter_type: Type of adapter to configure.
        webhook_url: URL to POST outcome events to (webhook adapter only).
        webhook_secret: Shared secret for HMAC signature verification.
        attribution_window_days: Days within which to attribute outcomes (default 90).
        enabled: Whether this adapter is active.
    """

    adapter_type: Literal["webhook", "manual"] = Field(
        ...,
        description="Type of outcome adapter to configure",
    )
    webhook_url: str | None = Field(
        default=None,
        description="URL for the webhook adapter to POST outcome events",
    )
    webhook_secret: str | None = Field(
        default=None,
        description="Shared HMAC secret for webhook signature verification",
    )
    attribution_window_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Days within which an outcome can be attributed to a decision",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this adapter is actively pulling outcome events",
    )


class OutcomeAdapterConfigResponse(BaseModel):
    """Response after configuring an outcome adapter.

    Attributes:
        tenant_id: Owning tenant.
        adapter_type: Configured adapter type.
        attribution_window_days: Configured attribution window.
        enabled: Whether the adapter is active.
        configured_at: Timestamp of configuration.
    """

    tenant_id: str
    adapter_type: str
    attribution_window_days: int
    enabled: bool
    configured_at: datetime
