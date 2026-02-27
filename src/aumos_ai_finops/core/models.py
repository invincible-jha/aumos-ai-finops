"""SQLAlchemy ORM models for the AumOS AI FinOps service.

All tables use the `fin_` prefix. Tenant-scoped tables extend AumOSModel
which supplies id (UUID), tenant_id, created_at, and updated_at columns.

Domain model:
  CostRecord             — GPU/CPU/storage cost allocations per tenant per time window
  TokenUsage             — Per-model, per-tenant token consumption records
  ROICalculation         — Multi-touch attribution ROI calculations
  Budget                 — Per-tenant budget thresholds with alert config
  BudgetAlert            — Triggered budget alerts (warning and critical)
  RoutingRecommendation  — Cost-optimized model routing suggestions

P0.2 chargeback models:
  CostAllocation         — Aggregated cost allocated to a team/project per period
  BudgetLimit            — Configurable spend limits with hard-cap enforcement
"""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel


class CostRecord(AumOSModel):
    """A cost allocation record for GPU, CPU, or storage resources per tenant.

    Captures the cost of AI infrastructure consumption for a specific time window,
    resource type, and workload. Populated from OpenCost/KubeCost API responses
    and direct provider billing data.

    Table: fin_cost_records
    """

    __tablename__ = "fin_cost_records"

    # Time window for this cost record
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Start of the billing period (UTC)",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="End of the billing period (UTC)",
    )

    # Resource classification
    resource_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="gpu | cpu | storage | network | memory",
    )
    resource_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Provider-specific resource identifier (node name, pod name, GPU device ID)",
    )
    gpu_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="GPU model: a100 | h100 | t4 | v100 (null for non-GPU resources)",
    )

    # Workload attribution
    workload_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Kubernetes workload or service name consuming this resource",
    )
    namespace: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Kubernetes namespace",
    )
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Reference to aumos-model-registry model (cross-service, no FK)",
    )

    # Cost values (USD)
    cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Total cost in USD for this resource in this period",
    )
    on_demand_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Cost if purchased at on-demand rates (for savings calculation)",
    )
    efficiency_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        comment="Resource utilization efficiency (0.0–1.0): actual_used / allocated",
    )

    # Data source
    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="opencost",
        comment="Data source: opencost | kubecost | manual | provider_api",
    )
    raw_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Raw response from cost provider for audit/debugging",
    )

    __table_args__ = (
        Index("ix_fin_cost_records_tenant_period", "tenant_id", "period_start", "period_end"),
        Index("ix_fin_cost_records_tenant_model", "tenant_id", "model_id"),
    )


class TokenUsage(AumOSModel):
    """Token consumption record for a model call or batch of calls per tenant.

    Tracks prompt tokens, completion tokens, and their associated costs.
    Fed by Kafka events from aumos-llm-serving on every model invocation.

    Table: fin_token_usage
    """

    __tablename__ = "fin_token_usage"

    # Model attribution
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Reference to aumos-model-registry model (cross-service, no FK)",
    )
    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Human-readable model name (e.g., gpt-4o, claude-opus-4, llama-3-70b)",
    )
    model_provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Model provider: openai | anthropic | google | self-hosted | huggingface",
    )
    model_tier: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="tier2",
        comment="Cost tier: tier1 (premium) | tier2 (mid) | tier3 (economy)",
    )

    # Time window
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Start of the aggregation window (UTC)",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the aggregation window (UTC)",
    )

    # Token counts
    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total prompt/input tokens consumed",
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total completion/output tokens generated",
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Sum of prompt + completion tokens",
    )
    request_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Number of model calls aggregated in this record",
    )

    # Costs (USD)
    prompt_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Cost of prompt tokens at provider pricing",
    )
    completion_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Cost of completion tokens at provider pricing",
    )
    total_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total token cost (prompt + completion)",
    )

    # Workload context
    workload_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Service or agent name that made these model calls",
    )
    use_case: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Business use case: code_generation | document_qa | classification | etc.",
    )

    __table_args__ = (
        Index("ix_fin_token_usage_tenant_model_period", "tenant_id", "model_id", "period_start"),
        Index("ix_fin_token_usage_tenant_period", "tenant_id", "period_start"),
    )


class ROICalculation(AumOSModel):
    """Multi-touch attribution ROI calculation for an AI initiative.

    Computes ROI as:
      (productivity_gain + quality_improvement + risk_avoidance - ai_cost) / ai_cost * 100

    Calculations are immutable once completed — create a new record to recalculate.

    Table: fin_roi_calculations
    """

    __tablename__ = "fin_roi_calculations"

    # Initiative metadata
    initiative_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Name of the AI initiative being measured",
    )
    initiative_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Type: code_assistant | document_qa | data_pipeline | customer_service | custom",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Detailed description of the initiative and measurement methodology",
    )

    # Measurement period
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Start of the ROI measurement period",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the ROI measurement period",
    )

    # Benefit components (USD)
    productivity_gain_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Hours saved × hourly_rate × headcount over the measurement period",
    )
    quality_improvement_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Error reduction rate × average error cost",
    )
    risk_avoidance_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Incidents prevented × average incident cost",
    )
    total_benefit_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Sum of all benefit components",
    )

    # Cost components (USD)
    gpu_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="GPU infrastructure cost from fin_cost_records",
    )
    token_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="LLM token cost from fin_token_usage",
    )
    infra_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Other infrastructure costs (CPU, storage, network)",
    )
    total_ai_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total AI investment cost for this period",
    )

    # ROI result
    roi_percent: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="ROI as percentage: (benefit - cost) / cost * 100",
    )
    payback_period_days: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Estimated days to break even on AI investment",
    )

    # Input assumptions (for audit and reproducibility)
    assumptions: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Input parameters: hours_saved, headcount, hourly_rate, error_reduction_rate, etc.",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="completed",
        comment="completed | draft | archived",
    )

    __table_args__ = (
        Index("ix_fin_roi_calculations_tenant_period", "tenant_id", "period_start"),
    )


class Budget(AumOSModel):
    """Per-tenant budget threshold for AI spending with alerting configuration.

    Supports monthly, quarterly, or annual budget periods. When actual spend
    crosses the configured threshold percentages, BudgetAlert records are created
    and finops.budget_exceeded events are published to Kafka.

    Table: fin_budgets
    """

    __tablename__ = "fin_budgets"

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Human-readable budget name (e.g., 'Q1 2026 AI Infrastructure')",
    )
    budget_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="monthly",
        comment="monthly | quarterly | annual | custom",
    )
    scope: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="all",
        comment="Budget scope: all | gpu | tokens | model_id:{uuid}",
    )

    # Budget period
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Start of the budget period",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the budget period",
    )

    # Amounts
    limit_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Maximum allowed spend in USD for this period",
    )
    warning_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.80,
        comment="Fraction of limit that triggers a warning alert (e.g., 0.80 = 80%)",
    )
    critical_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.95,
        comment="Fraction of limit that triggers a critical alert (e.g., 0.95 = 95%)",
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether this budget is actively monitored",
    )
    notification_channels: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Notification targets: [{type: email, address: ...}, {type: slack, webhook: ...}]",
    )

    # Relationships
    alerts: Mapped[list["BudgetAlert"]] = relationship(
        "BudgetAlert",
        back_populates="budget",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", "period_start", name="uq_fin_budgets_tenant_name_period"),
    )


class BudgetAlert(AumOSModel):
    """A triggered budget alert record for audit and notification tracking.

    Created when actual spend crosses a warning or critical threshold.
    Tracks whether the alert has been acknowledged.

    Table: fin_budget_alerts
    """

    __tablename__ = "fin_budget_alerts"

    budget_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fin_budgets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the parent budget",
    )
    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="warning | critical | info",
    )
    actual_spend_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Actual spend at the time the alert was triggered",
    )
    threshold_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Dollar threshold that was crossed (limit_usd × threshold_fraction)",
    )
    utilization_percent: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Percentage of budget consumed: actual_spend / limit * 100",
    )
    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Human-readable alert message",
    )
    acknowledged: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this alert has been acknowledged by a user",
    )
    acknowledged_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User ID who acknowledged the alert (cross-service, no FK)",
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the alert was acknowledged",
    )
    kafka_event_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Kafka event correlation ID for the finops.budget_exceeded event",
    )

    # Relationships
    budget: Mapped["Budget"] = relationship("Budget", back_populates="alerts")

    __table_args__ = (
        Index("ix_fin_budget_alerts_budget_severity", "budget_id", "severity"),
        Index("ix_fin_budget_alerts_tenant_acknowledged", "tenant_id", "acknowledged"),
    )


class RoutingRecommendation(AumOSModel):
    """A cost-optimized model routing recommendation for a workload.

    Balances cost, quality, and latency objectives using weighted scoring to
    recommend which model to use for a given workload context.

    Table: fin_routing_recommendations
    """

    __tablename__ = "fin_routing_recommendations"

    # Workload context
    workload_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Service or agent requesting routing guidance",
    )
    use_case: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Business use case: code_generation | document_qa | classification | summarization | etc.",
    )
    quality_requirement: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="standard",
        comment="Quality threshold: economy | standard | premium",
    )
    latency_requirement_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Maximum acceptable latency in milliseconds (null = no strict requirement)",
    )
    estimated_monthly_requests: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1000,
        comment="Estimated monthly request volume for cost projection",
    )
    avg_prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=500,
        comment="Average prompt tokens per request",
    )
    avg_completion_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=200,
        comment="Average completion tokens per request",
    )

    # Recommendation output
    recommended_model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="Recommended model from aumos-model-registry (cross-service, no FK)",
    )
    recommended_model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable recommended model name",
    )
    fallback_model_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Fallback model if primary is unavailable",
    )
    routing_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Composite score (0.0–1.0) balancing cost, quality, and latency",
    )
    estimated_monthly_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Projected monthly cost for the recommended model",
    )
    estimated_savings_vs_premium_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Estimated monthly savings compared to the most expensive viable option",
    )
    reasoning: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable explanation of the routing recommendation",
    )
    candidate_models: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="All evaluated models with their scores: [{model_name, cost_score, quality_score, ...}]",
    )

    __table_args__ = (
        Index("ix_fin_routing_tenant_workload", "tenant_id", "workload_name"),
    )


# ---------------------------------------------------------------------------
# P0.2 Chargeback models
# ---------------------------------------------------------------------------


class CostAllocation(AumOSModel):
    """Aggregated AI cost allocation for a team and project within a period.

    Records rolled-up token, inference, and storage costs to support per-team
    chargeback reporting and multi-period trend analysis.

    Table: fin_cost_allocations
    """

    __tablename__ = "fin_cost_allocations"

    # Scope identifiers
    team_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Team identifier within the tenant",
    )
    project_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Project identifier within the team",
    )

    # Billing period
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Start of the cost allocation period (UTC)",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the cost allocation period (UTC)",
    )

    # Service and model classification
    service: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Service that incurred the cost (e.g., aumos-llm-serving)",
    )
    model_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="",
        comment="Model identifier (e.g., gpt-4o, claude-opus-4, or empty for non-LLM costs)",
    )

    # Token counters (LLM costs)
    total_input_tokens: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        comment="Total prompt/input tokens consumed in this period",
    )
    total_output_tokens: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        comment="Total completion/output tokens generated in this period",
    )
    total_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total allocated cost in USD for this period",
    )

    # Non-token cost components
    inference_minutes: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total compute time in minutes for inference (non-token compute charges)",
    )
    storage_gb_days: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Storage consumed in GB-days (vector stores, model artifacts, etc.)",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "team_id",
            "project_id",
            "service",
            "model_id",
            "period_start",
            name="uq_fin_cost_allocations_scope_period",
        ),
        Index(
            "ix_fin_cost_allocations_tenant_team_period",
            "tenant_id",
            "team_id",
            "period_start",
        ),
        Index(
            "ix_fin_cost_allocations_tenant_project_period",
            "tenant_id",
            "project_id",
            "period_start",
        ),
    )


class BudgetLimit(AumOSModel):
    """A spending limit for a tenant/team combination with alerting configuration.

    Supports monthly, quarterly, and annual periods. When actual spend crosses
    the alert threshold, a warning event is emitted. When hard_cap is enabled
    and the limit is reached, requests that would exceed the budget are blocked.

    Table: fin_budget_limits
    """

    __tablename__ = "fin_budget_limits"

    # Scope — team_id is optional to allow tenant-wide limits
    team_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Team scope for this limit (null = applies to entire tenant)",
    )

    # Period type
    period_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="monthly",
        comment="Billing period granularity: monthly | quarterly | annual",
    )

    # Limit configuration
    limit_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Maximum allowed spend in USD for the configured period",
    )
    alert_threshold_pct: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=80,
        comment="Percentage of limit that triggers a warning alert (e.g., 80 = 80%)",
    )
    hard_cap: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="When True, block requests that would cause spend to exceed limit_usd",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether this budget limit is actively enforced",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "team_id",
            "period_type",
            name="uq_fin_budget_limits_tenant_team_period",
        ),
        Index("ix_fin_budget_limits_tenant_active", "tenant_id", "is_active"),
    )


# ---------------------------------------------------------------------------
# P3.1 Cost-to-Outcome Attribution models
# ---------------------------------------------------------------------------


class CostOutcomeAttribution(AumOSModel):
    """Per-decision cost breakdown joined with business outcome for ROI measurement.

    Records the six cost components for every AI decision and, once attributed,
    the business outcome value and computed ROI. The 90-day attribution window
    is enforced at the service layer.

    All monetary columns use DECIMAL(18,6) for precision — never Float.

    Table: fin_cost_outcome_attributions
    """

    __tablename__ = "fin_cost_outcome_attributions"

    # Decision identity
    decision_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique identifier for the AI decision (UUID string from the AI system)",
    )
    ai_system_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Identifier of the AI system that produced the decision",
    )
    use_case: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Business use case label (e.g., fraud_detection, claims_triage)",
    )
    decision_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Timestamp when the AI decision was made (UTC)",
    )

    # Six cost components (USD, DECIMAL for precision)
    input_token_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="Cost of prompt/input tokens in USD",
    )
    output_token_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="Cost of completion/output tokens in USD",
    )
    compute_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="GPU/CPU compute cost during inference in USD",
    )
    storage_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="Storage cost attributed to this decision in USD",
    )
    egress_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="Network egress cost for delivering the response in USD",
    )
    human_review_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="Human review cost if a human reviewed this decision in USD (0 if no review)",
    )

    # Derived total (sum of all six components)
    total_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="Sum of all six cost components in USD",
    )

    # Business outcome (populated when attribution is complete)
    outcome_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Categorical outcome type (e.g., revenue_generated, cost_saved, risk_avoided)",
    )
    outcome_value_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 6),
        nullable=True,
        comment="Monetized business outcome value in USD",
    )
    roi_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4),
        nullable=True,
        comment="ROI = (outcome_value - total_cost) / total_cost * 100",
    )
    attribution_complete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="True once an outcome has been matched to this decision",
    )
    outcome_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the business outcome was realized (UTC)",
    )

    __table_args__ = (
        Index(
            "ix_fin_coa_tenant_use_case_decision",
            "tenant_id",
            "use_case",
            "decision_at",
        ),
        Index(
            "ix_fin_coa_tenant_complete",
            "tenant_id",
            "attribution_complete",
        ),
    )


class ROISummary(AumOSModel):
    """Aggregated ROI summary for a use case over a reporting period.

    Pre-aggregated to support fast dashboard queries without scanning
    individual attribution records. Refreshed daily by a background job.

    Table: fin_roi_summaries
    """

    __tablename__ = "fin_roi_summaries"

    use_case: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Business use case this summary covers",
    )
    period_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Date of this summary record (UTC, truncated to day)",
    )
    total_decisions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total AI decisions processed for this use case in this period",
    )
    total_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="Sum of all decision costs for this use case and period",
    )
    total_outcome_value_usd: Mapped[Decimal] = mapped_column(
        Numeric(18, 6),
        nullable=False,
        default=Decimal("0"),
        comment="Sum of all attributed outcome values for this use case and period",
    )
    avg_roi_pct: Mapped[Decimal] = mapped_column(
        Numeric(10, 4),
        nullable=False,
        default=Decimal("0"),
        comment="Average ROI across decisions that have attributed outcomes",
    )
    decisions_with_outcome: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Count of decisions that have a matched outcome",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "use_case",
            "period_date",
            name="uq_fin_roi_summaries_tenant_usecase_period",
        ),
        Index(
            "ix_fin_roi_summaries_tenant_period",
            "tenant_id",
            "period_date",
        ),
    )
