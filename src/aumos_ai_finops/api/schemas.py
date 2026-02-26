"""Pydantic request and response schemas for the AumOS AI FinOps API.

All API inputs and outputs are typed Pydantic models — never raw dicts.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / embedded schemas
# ---------------------------------------------------------------------------


class CostRecordResponse(BaseModel):
    """Response schema for a single cost record."""

    id: uuid.UUID
    tenant_id: str
    resource_type: str
    resource_id: str
    gpu_type: str | None
    workload_name: str | None
    namespace: str | None
    model_id: uuid.UUID | None
    cost_usd: float
    on_demand_cost_usd: float
    efficiency_rate: float
    period_start: datetime
    period_end: datetime
    source: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenUsageResponse(BaseModel):
    """Response schema for a single token usage record."""

    id: uuid.UUID
    tenant_id: str
    model_id: uuid.UUID
    model_name: str
    model_provider: str
    model_tier: str
    period_start: datetime
    period_end: datetime
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_count: int
    prompt_cost_usd: float
    completion_cost_usd: float
    total_cost_usd: float
    workload_name: str | None
    use_case: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelTokenAggregateResponse(BaseModel):
    """Aggregated token usage summary for a single model."""

    model_name: str
    model_provider: str
    model_tier: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    request_count: int


class ROICalculationResponse(BaseModel):
    """Response schema for a completed ROI calculation."""

    id: uuid.UUID
    tenant_id: str
    initiative_name: str
    initiative_type: str
    description: str | None
    period_start: datetime
    period_end: datetime
    productivity_gain_usd: float
    quality_improvement_usd: float
    risk_avoidance_usd: float
    total_benefit_usd: float
    gpu_cost_usd: float
    token_cost_usd: float
    infra_cost_usd: float
    total_ai_cost_usd: float
    roi_percent: float
    payback_period_days: int | None
    assumptions: dict[str, Any]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class BudgetResponse(BaseModel):
    """Response schema for a budget threshold."""

    id: uuid.UUID
    tenant_id: str
    name: str
    budget_type: str
    scope: str
    period_start: datetime
    period_end: datetime
    limit_usd: float
    warning_threshold: float
    critical_threshold: float
    is_active: bool
    notification_channels: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BudgetAlertResponse(BaseModel):
    """Response schema for a budget alert."""

    id: uuid.UUID
    tenant_id: str
    budget_id: uuid.UUID
    severity: str
    actual_spend_usd: float
    threshold_usd: float
    utilization_percent: float
    message: str
    acknowledged: bool
    acknowledged_by: uuid.UUID | None
    acknowledged_at: datetime | None
    kafka_event_id: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class RoutingRecommendationResponse(BaseModel):
    """Response schema for a routing recommendation."""

    id: uuid.UUID
    tenant_id: str
    workload_name: str
    use_case: str
    quality_requirement: str
    latency_requirement_ms: int | None
    estimated_monthly_requests: int
    avg_prompt_tokens: int
    avg_completion_tokens: int
    recommended_model_name: str
    fallback_model_name: str | None
    routing_score: float
    estimated_monthly_cost_usd: float
    estimated_savings_vs_premium_usd: float
    reasoning: str | None
    candidate_models: list[dict[str, Any]]
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Cost endpoint schemas
# ---------------------------------------------------------------------------


class CostQueryParams(BaseModel):
    """Query parameters for cost breakdown endpoints."""

    period_start: datetime = Field(description="Start of the cost query window (UTC)")
    period_end: datetime = Field(description="End of the cost query window (UTC)")
    resource_type: str | None = Field(
        default=None,
        description="Filter by resource type: gpu | cpu | storage | network | memory",
    )


class RecordCostRequest(BaseModel):
    """Request body for manually recording a cost entry."""

    resource_type: str = Field(
        ...,
        pattern="^(gpu|cpu|storage|network|memory)$",
        description="Type of resource: gpu | cpu | storage | network | memory",
    )
    resource_id: str = Field(
        ...,
        max_length=255,
        description="Provider-specific resource identifier",
    )
    cost_usd: float = Field(
        ...,
        gt=0,
        description="Total cost in USD (must be positive)",
    )
    period_start: datetime = Field(description="Start of the billing window (UTC)")
    period_end: datetime = Field(description="End of the billing window (UTC)")
    gpu_type: str | None = Field(
        default=None,
        pattern="^(a100|h100|t4|v100|a10|l4)?$",
        description="GPU model type (required when resource_type=gpu)",
    )
    workload_name: str | None = Field(default=None, max_length=255)
    namespace: str | None = Field(default=None, max_length=255)
    model_id: uuid.UUID | None = Field(default=None, description="Associated model registry ID")
    on_demand_cost_usd: float = Field(default=0.0, ge=0)
    efficiency_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(
        default="manual",
        pattern="^(opencost|kubecost|manual|provider_api)$",
    )


class SyncOpenCostRequest(BaseModel):
    """Request body for syncing cost data from OpenCost."""

    namespace: str | None = Field(default=None, description="Kubernetes namespace filter (None = all)")
    window: str = Field(
        default="24h",
        description="OpenCost time window: 24h | 7d | 30d | or ISO date range",
    )


# ---------------------------------------------------------------------------
# Token endpoint schemas
# ---------------------------------------------------------------------------


class RecordTokenUsageRequest(BaseModel):
    """Request body for recording token consumption."""

    model_id: uuid.UUID = Field(description="Model registry UUID")
    model_name: str = Field(..., max_length=255, description="Human-readable model name")
    model_provider: str = Field(
        ...,
        max_length=100,
        description="Provider: openai | anthropic | google | self-hosted | huggingface",
    )
    model_tier: str = Field(
        default="tier2",
        pattern="^(tier1|tier2|tier3)$",
        description="Cost tier: tier1 (premium) | tier2 (mid) | tier3 (economy)",
    )
    period_start: datetime
    period_end: datetime
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    request_count: int = Field(default=1, ge=1)
    workload_name: str | None = Field(default=None, max_length=255)
    use_case: str | None = Field(default=None, max_length=100)


# ---------------------------------------------------------------------------
# ROI endpoint schemas
# ---------------------------------------------------------------------------


class ROICalculateRequest(BaseModel):
    """Request body for calculating ROI for an AI initiative."""

    initiative_name: str = Field(
        ...,
        min_length=3,
        max_length=255,
        description="Human-readable name of the AI initiative",
        examples=["Code Assistant for Engineering Team"],
    )
    initiative_type: str = Field(
        ...,
        max_length=100,
        description="Type: code_assistant | document_qa | data_pipeline | customer_service | custom",
        examples=["code_assistant"],
    )
    period_start: datetime = Field(description="Start of the measurement period")
    period_end: datetime = Field(description="End of the measurement period")
    hours_saved: float = Field(
        ...,
        ge=0,
        description="Total hours of human labor saved by the AI initiative",
    )
    headcount: int = Field(
        ...,
        ge=1,
        description="Number of team members benefiting from the initiative",
    )
    hourly_rate_usd: float | None = Field(
        default=None,
        gt=0,
        description="Fully-loaded hourly rate in USD (defaults to service setting ~$75)",
    )
    error_reduction_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of errors eliminated by the AI initiative (0.0–1.0)",
    )
    avg_error_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Average cost per error before AI (USD)",
    )
    incidents_prevented: int = Field(
        default=0,
        ge=0,
        description="Number of incidents prevented by the AI initiative",
    )
    avg_incident_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Average cost per incident (USD)",
    )
    additional_infra_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Additional infrastructure costs not captured in cost records",
    )
    description: str | None = Field(
        default=None,
        description="Description of the initiative and measurement methodology",
    )


# ---------------------------------------------------------------------------
# Budget endpoint schemas
# ---------------------------------------------------------------------------


class CreateBudgetRequest(BaseModel):
    """Request body for creating a new budget threshold."""

    name: str = Field(
        ...,
        min_length=3,
        max_length=255,
        description="Human-readable budget name",
        examples=["Q1 2026 GPU Infrastructure"],
    )
    limit_usd: float = Field(
        ...,
        gt=0,
        description="Maximum spend allowed in USD",
    )
    period_start: datetime = Field(description="Budget period start (UTC)")
    period_end: datetime = Field(description="Budget period end (UTC)")
    budget_type: str = Field(
        default="monthly",
        pattern="^(monthly|quarterly|annual|custom)$",
    )
    scope: str = Field(
        default="all",
        description="Budget scope: all | gpu | tokens | model_id:{uuid}",
    )
    warning_threshold: float | None = Field(
        default=None,
        ge=0.5,
        le=1.0,
        description="Warning alert threshold as fraction of limit (e.g., 0.80 = 80%)",
    )
    critical_threshold: float | None = Field(
        default=None,
        ge=0.5,
        le=1.0,
        description="Critical alert threshold as fraction of limit (e.g., 0.95 = 95%)",
    )
    notification_channels: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Alert destinations: [{type: email, address: ...}, {type: slack, webhook: ...}]",
    )


# ---------------------------------------------------------------------------
# Routing endpoint schemas
# ---------------------------------------------------------------------------


class RoutingOptimizeRequest(BaseModel):
    """Request body for cost-optimized model routing recommendation."""

    workload_name: str = Field(
        ...,
        max_length=255,
        description="Service or agent requesting routing guidance",
        examples=["code-review-agent"],
    )
    use_case: str = Field(
        ...,
        max_length=100,
        description="Business use case: code_generation | document_qa | classification | summarization",
        examples=["code_generation"],
    )
    quality_requirement: str = Field(
        default="standard",
        pattern="^(economy|standard|premium)$",
        description="Required quality level: economy | standard | premium",
    )
    latency_requirement_ms: int | None = Field(
        default=None,
        gt=0,
        description="Maximum acceptable latency in milliseconds (None = no constraint)",
    )
    estimated_monthly_requests: int = Field(
        default=1000,
        gt=0,
        description="Estimated monthly request volume for cost projection",
    )
    avg_prompt_tokens: int = Field(
        default=500,
        gt=0,
        description="Average prompt tokens per request",
    )
    avg_completion_tokens: int = Field(
        default=200,
        gt=0,
        description="Average completion tokens per request",
    )


# ---------------------------------------------------------------------------
# Dashboard schemas
# ---------------------------------------------------------------------------


class DashboardSummaryResponse(BaseModel):
    """Executive dashboard data aggregating cost, token, and ROI metrics."""

    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_cost_usd: float
    gpu_cost_usd: float
    token_cost_usd: float
    infra_cost_usd: float
    total_tokens: int
    total_requests: int
    active_budgets: int
    budget_utilization_percent: float | None
    latest_roi_percent: float | None
    cost_trend_percent: float | None  # Positive = costs rising vs prior period
    top_models_by_cost: list[ModelTokenAggregateResponse]
    unacknowledged_alerts: int
