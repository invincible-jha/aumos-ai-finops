"""Additional SQLAlchemy ORM models for AI FinOps gap features.

New models (Gaps #262-267):
    FinFocusExport         — FOCUS 1.3 export job records (GAP-262)
    FinExternalLLMCost     — external LLM provider cost records (GAP-264)
    FinCostAnomalyBaseline — rolling baseline for cost anomaly detection (GAP-265)
    FinCostAnomaly         — detected cost anomaly records (GAP-265)
    FinChargebackRecord    — chargeback allocation records per cost centre (GAP-266)
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from aumos_common.database import AumOSModel


class FinFocusExport(AumOSModel):
    """Record of a FOCUS 1.3 cost data export job.

    Each export job request creates one record. The export is generated
    asynchronously and stored as a downloadable artifact.

    Table: fin_focus_exports
    """

    __tablename__ = "fin_focus_exports"

    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Start of the billing period covered by this export",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the billing period covered by this export",
    )
    format: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="csv",
        comment="Export format: csv | jsonlines",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        comment="Export status: pending | generating | completed | failed",
    )
    row_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of FOCUS rows in the completed export",
    )
    download_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="Pre-signed download URL for the completed export file",
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Size of the generated export file in bytes",
    )
    error_detail: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if export generation failed",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the export was completed",
    )
    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User UUID who requested this export",
    )


class FinExternalLLMCost(AumOSModel):
    """Cost record from an external LLM provider API.

    Stores normalized usage and cost data fetched from provider billing APIs
    (Anthropic, OpenAI, Google Vertex AI). Used alongside self-hosted GPU
    costs for unified AI spend visibility.

    Table: fin_external_llm_costs
    """

    __tablename__ = "fin_external_llm_costs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "model_id",
            "period_start",
            name="uq_fin_external_llm_costs_tenant_provider_model_period",
        ),
    )

    provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="LLM provider name: anthropic | openai | google | azure_openai",
    )
    model_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Provider-qualified model ID (e.g., anthropic/claude-opus-4-6)",
    )
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Start of the billing period this record covers",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the billing period this record covers",
    )
    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total input/prompt tokens consumed in this period",
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total output/completion tokens consumed in this period",
    )
    total_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total cost in USD as reported by the provider",
    )
    request_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of API requests in this period",
    )
    raw_provider_data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Raw provider API response for audit and reconciliation",
    )
    fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when this data was fetched from the provider API",
    )


class FinCostAnomalyBaseline(AumOSModel):
    """Rolling baseline statistics for cost anomaly detection.

    Stores per-tenant, per-metric baseline statistics used to identify
    anomalous spending patterns using z-score analysis.

    Table: fin_cost_anomaly_baselines
    """

    __tablename__ = "fin_cost_anomaly_baselines"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "metric_key",
            name="uq_fin_cost_anomaly_baselines_tenant_metric",
        ),
    )

    metric_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Metric identifier: 'total' or 'model:{model_id}'",
    )
    mean_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Rolling mean daily cost in USD",
    )
    stddev_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Rolling standard deviation of daily cost",
    )
    sample_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of observations used in baseline computation",
    )
    last_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of the last baseline update",
    )
    window_days: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=30,
        comment="Rolling window in days for baseline computation",
    )


class FinCostAnomaly(AumOSModel):
    """Record of a detected cost anomaly event.

    Created when the CostAnomalyDetector identifies a cost pattern that
    exceeds the configured z-score threshold. Used for alerting, audit
    trail, and trend analysis.

    Table: fin_cost_anomalies
    """

    __tablename__ = "fin_cost_anomalies"

    model_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Model ID with the anomaly (None for aggregate tenant anomaly)",
    )
    current_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Observed cost that triggered the anomaly signal",
    )
    baseline_mean_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Rolling baseline mean at the time of detection",
    )
    baseline_stddev_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Rolling baseline standard deviation at detection time",
    )
    z_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Z-score of the anomaly (standard deviations above mean)",
    )
    percent_above_baseline: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Cost increase as a percentage above the baseline mean",
    )
    severity: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Anomaly severity: warning | critical",
    )
    resolved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True when the anomaly has been acknowledged and resolved",
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the anomaly was marked as resolved",
    )
    resolution_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Admin notes explaining the root cause or resolution",
    )


class FinChargebackRecord(AumOSModel):
    """Chargeback allocation record per cost centre or business unit.

    Maps AI infrastructure costs to internal cost centres for showback
    and chargeback reporting. Enables departmental accountability for
    AI spending in enterprise environments.

    Table: fin_chargeback_records
    """

    __tablename__ = "fin_chargeback_records"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "cost_centre_id",
            "period_start",
            name="uq_fin_chargeback_records_tenant_centre_period",
        ),
    )

    cost_centre_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Internal cost centre or business unit identifier",
    )
    cost_centre_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable cost centre display name",
    )
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Start of the chargeback period",
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the chargeback period",
    )
    allocated_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Total AI cost allocated to this cost centre for the period",
    )
    allocation_method: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="proportional",
        comment="Allocation method: proportional | fixed | usage_based | equal_split",
    )
    allocation_basis: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Allocation basis details: {model_ids, usage_weight, token_share, etc.}",
    )
    invoice_number: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Internal invoice or charge reference number",
    )
    approved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True when the chargeback allocation has been approved by finance",
    )
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the chargeback was approved",
    )
    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User UUID who approved the chargeback allocation",
    )


__all__ = [
    "FinFocusExport",
    "FinExternalLLMCost",
    "FinCostAnomalyBaseline",
    "FinCostAnomaly",
    "FinChargebackRecord",
]
