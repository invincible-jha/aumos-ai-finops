"""Create fin_cost_outcome_attributions and fin_roi_summaries tables.

Revision ID: 002_fin_attribution
Revises: 001_fin_initial
Create Date: 2026-02-26

P3.1: AI Cost-to-Outcome Attribution tables.
  - fin_cost_outcome_attributions — per-decision cost + outcome + ROI
  - fin_roi_summaries — pre-aggregated use-case ROI summaries

All monetary columns are NUMERIC(18,6) — never FLOAT.
Both tables have tenant_id for RLS enforcement.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "002_fin_attribution"
down_revision = "001_fin_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create attribution tables with RLS policies and indexes."""

    # fin_cost_outcome_attributions
    op.create_table(
        "fin_cost_outcome_attributions",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Decision identity
        sa.Column(
            "decision_id",
            sa.String(255),
            nullable=False,
            unique=True,
            comment="Unique identifier of the AI decision (UUID string from the AI system)",
        ),
        sa.Column(
            "ai_system_id",
            sa.String(255),
            nullable=False,
            comment="Identifier of the AI system that produced the decision",
        ),
        sa.Column(
            "use_case",
            sa.String(255),
            nullable=False,
            comment="Business use case label",
        ),
        sa.Column(
            "decision_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Timestamp when the AI decision was made (UTC)",
        ),
        # Six cost components (DECIMAL for precision)
        sa.Column(
            "input_token_cost",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="Cost of prompt/input tokens in USD",
        ),
        sa.Column(
            "output_token_cost",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="Cost of completion/output tokens in USD",
        ),
        sa.Column(
            "compute_cost",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="GPU/CPU compute cost during inference in USD",
        ),
        sa.Column(
            "storage_cost",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="Storage cost attributed to this decision in USD",
        ),
        sa.Column(
            "egress_cost",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="Network egress cost for response delivery in USD",
        ),
        sa.Column(
            "human_review_cost",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="Human review cost in USD (0 if no review required)",
        ),
        sa.Column(
            "total_cost_usd",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="Sum of all six cost components in USD",
        ),
        # Outcome fields
        sa.Column(
            "outcome_type",
            sa.String(100),
            nullable=True,
            comment="Categorical outcome type (e.g., revenue_generated, cost_saved)",
        ),
        sa.Column(
            "outcome_value_usd",
            sa.Numeric(18, 6),
            nullable=True,
            comment="Monetized business outcome value in USD",
        ),
        sa.Column(
            "roi_pct",
            sa.Numeric(10, 4),
            nullable=True,
            comment="ROI = (outcome_value - total_cost) / total_cost * 100",
        ),
        sa.Column(
            "attribution_complete",
            sa.Boolean,
            nullable=False,
            server_default="false",
            comment="True once an outcome has been matched to this decision",
        ),
        sa.Column(
            "outcome_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when the business outcome was realized (UTC)",
        ),
    )
    op.create_index(
        "ix_fin_coa_decision_id",
        "fin_cost_outcome_attributions",
        ["decision_id"],
        unique=True,
    )
    op.create_index(
        "ix_fin_coa_tenant_use_case_decision",
        "fin_cost_outcome_attributions",
        ["tenant_id", "use_case", "decision_at"],
    )
    op.create_index(
        "ix_fin_coa_tenant_complete",
        "fin_cost_outcome_attributions",
        ["tenant_id", "attribution_complete"],
    )
    op.create_index(
        "ix_fin_coa_ai_system",
        "fin_cost_outcome_attributions",
        ["ai_system_id"],
    )

    op.execute("ALTER TABLE fin_cost_outcome_attributions ENABLE ROW LEVEL SECURITY;")
    op.execute(
        "CREATE POLICY fin_coa_tenant_isolation ON fin_cost_outcome_attributions "
        "USING (tenant_id = current_setting('app.current_tenant', true));"
    )

    # fin_roi_summaries
    op.create_table(
        "fin_roi_summaries",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "use_case",
            sa.String(255),
            nullable=False,
            comment="Business use case this summary covers",
        ),
        sa.Column(
            "period_date",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Date of this summary (UTC, truncated to day)",
        ),
        sa.Column(
            "total_decisions",
            sa.Integer,
            nullable=False,
            server_default="0",
            comment="Total decisions processed in this period",
        ),
        sa.Column(
            "total_cost_usd",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="Sum of all decision costs",
        ),
        sa.Column(
            "total_outcome_value_usd",
            sa.Numeric(18, 6),
            nullable=False,
            server_default="0",
            comment="Sum of all attributed outcome values",
        ),
        sa.Column(
            "avg_roi_pct",
            sa.Numeric(10, 4),
            nullable=False,
            server_default="0",
            comment="Average ROI across decisions with outcomes",
        ),
        sa.Column(
            "decisions_with_outcome",
            sa.Integer,
            nullable=False,
            server_default="0",
            comment="Count of decisions with a matched outcome",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "use_case",
            "period_date",
            name="uq_fin_roi_summaries_tenant_usecase_period",
        ),
    )
    op.create_index(
        "ix_fin_roi_summaries_tenant_period",
        "fin_roi_summaries",
        ["tenant_id", "period_date"],
    )
    op.create_index(
        "ix_fin_roi_summaries_use_case",
        "fin_roi_summaries",
        ["use_case"],
    )

    op.execute("ALTER TABLE fin_roi_summaries ENABLE ROW LEVEL SECURITY;")
    op.execute(
        "CREATE POLICY fin_roi_summaries_tenant_isolation ON fin_roi_summaries "
        "USING (tenant_id = current_setting('app.current_tenant', true));"
    )


def downgrade() -> None:
    """Drop attribution tables."""
    op.drop_table("fin_roi_summaries")
    op.drop_table("fin_cost_outcome_attributions")
