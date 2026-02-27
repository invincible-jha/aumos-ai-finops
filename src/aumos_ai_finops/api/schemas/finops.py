"""Pydantic request/response schemas for the chargeback and budget-limit APIs.

These schemas extend the existing finops API surface with chargeback reporting
and the new BudgetLimit configuration endpoints.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Cost breakdown schemas
# ---------------------------------------------------------------------------


class CostLineItem(BaseModel):
    """A single cost line item within a breakdown report.

    Attributes:
        group_key: The value of the group_by dimension (e.g., the team_id, model name).
        dimension: The grouping dimension (team | project | model | service).
        total_cost_usd: Total cost in USD for this group.
        total_input_tokens: Aggregated prompt tokens (0 for non-LLM costs).
        total_output_tokens: Aggregated completion tokens.
        request_count: Number of requests contributing to this line.
        period_start: Start of the report period.
        period_end: End of the report period.
    """

    group_key: str
    dimension: str
    total_cost_usd: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_count: int = 0
    period_start: datetime
    period_end: datetime


class CostBreakdownResponse(BaseModel):
    """Response for the cost breakdown endpoint.

    Attributes:
        tenant_id: The tenant this breakdown belongs to.
        period_start: Report period start (UTC).
        period_end: Report period end (UTC).
        group_by: The grouping dimension applied.
        total_cost_usd: Grand total cost across all groups.
        line_items: Individual cost line items sorted by cost descending.
    """

    tenant_id: str
    period_start: datetime
    period_end: datetime
    group_by: str
    total_cost_usd: float
    line_items: list[CostLineItem]


# ---------------------------------------------------------------------------
# Budget limit schemas
# ---------------------------------------------------------------------------


class BudgetCreateRequest(BaseModel):
    """Request body for creating a new budget limit.

    Attributes:
        team_id: Optional team scope. Omit for a tenant-wide limit.
        period_type: Billing period granularity.
        limit_usd: Maximum spend allowed per period.
        alert_threshold_pct: Percentage of limit that triggers a warning alert.
        hard_cap: When True, block requests that would cause spend to exceed limit.
    """

    team_id: str | None = Field(
        default=None,
        description="Team scope (null = tenant-wide limit)",
    )
    period_type: Literal["monthly", "quarterly", "annual"] = Field(
        default="monthly",
        description="Billing period granularity",
    )
    limit_usd: float = Field(
        ...,
        gt=0,
        description="Maximum allowed spend in USD per period",
    )
    alert_threshold_pct: int = Field(
        default=80,
        ge=1,
        le=100,
        description="Percentage of limit that triggers a warning alert (1–100)",
    )
    hard_cap: bool = Field(
        default=False,
        description="Block requests when spend would exceed limit",
    )


class BudgetStatusResponse(BaseModel):
    """Current status of a budget limit for a team or tenant.

    Attributes:
        budget_limit_id: UUID of the BudgetLimit record.
        tenant_id: Tenant this budget belongs to.
        team_id: Team scope (None = tenant-wide).
        period_type: Billing period granularity.
        limit_usd: Configured spend limit.
        consumed_usd: Actual spend in the current period.
        remaining_usd: Remaining budget (may be negative when over-limit).
        utilization_pct: Consumed / limit * 100.
        alert_threshold_pct: Alert trigger percentage.
        hard_cap: Whether hard-cap enforcement is active.
        is_active: Whether this limit is actively enforced.
        alert_triggered: True when utilization_pct >= alert_threshold_pct.
        period_start: Start of the current billing period.
        period_end: End of the current billing period.
        projected_overrun_usd: Estimated overrun based on daily burn rate.
    """

    budget_limit_id: uuid.UUID
    tenant_id: str
    team_id: str | None
    period_type: str
    limit_usd: float
    consumed_usd: float
    remaining_usd: float
    utilization_pct: float
    alert_threshold_pct: int
    hard_cap: bool
    is_active: bool
    alert_triggered: bool
    period_start: datetime
    period_end: datetime
    projected_overrun_usd: float | None = None


class BudgetLimitResponse(BaseModel):
    """Response schema for a persisted BudgetLimit record.

    Attributes:
        id: UUID primary key.
        tenant_id: The owning tenant.
        team_id: Team scope (None = tenant-wide).
        period_type: Billing period granularity.
        limit_usd: Maximum spend limit.
        alert_threshold_pct: Warning alert threshold percentage.
        hard_cap: Whether hard-cap enforcement is enabled.
        is_active: Whether this limit is currently active.
        created_at: Record creation timestamp.
        updated_at: Record last-updated timestamp.
    """

    id: uuid.UUID
    tenant_id: str
    team_id: str | None
    period_type: str
    limit_usd: float
    alert_threshold_pct: int
    hard_cap: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Chargeback report schemas
# ---------------------------------------------------------------------------


class ChargebackLineItem(BaseModel):
    """A single team/project cost entry in a chargeback report.

    Attributes:
        team_id: Team identifier.
        project_id: Project identifier within the team.
        service: Service that incurred the cost.
        model_id: Model identifier (empty for non-LLM costs).
        total_input_tokens: Aggregated prompt tokens.
        total_output_tokens: Aggregated completion tokens.
        total_cost_usd: Total cost allocated to this team/project.
        inference_minutes: Compute minutes (non-token costs).
        storage_gb_days: Storage in GB-days.
    """

    team_id: str
    project_id: str
    service: str
    model_id: str
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    inference_minutes: float
    storage_gb_days: float


class ChargebackReportResponse(BaseModel):
    """Full chargeback report for a billing period.

    Attributes:
        tenant_id: The tenant this report covers.
        period_start: Report period start (UTC).
        period_end: Report period end (UTC).
        generated_at: Timestamp when this report was generated.
        total_cost_usd: Grand total cost across all teams/projects.
        line_items: Individual team/project cost entries.
        format: Output format requested (json | csv | pdf).
        csv_data: CSV string representation (only present when format=csv).
    """

    tenant_id: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    total_cost_usd: float
    line_items: list[ChargebackLineItem]
    format: str = "json"
    csv_data: str | None = None
