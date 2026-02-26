"""Create initial fin_ schema tables.

Revision ID: 001_fin_initial
Revises: —
Create Date: 2026-02-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001_fin_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all fin_ tables with RLS policies."""

    # fin_cost_records
    op.create_table(
        "fin_cost_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=False),
        sa.Column("resource_id", sa.String(255), nullable=False),
        sa.Column("gpu_type", sa.String(50), nullable=True),
        sa.Column("workload_name", sa.String(255), nullable=True),
        sa.Column("namespace", sa.String(255), nullable=True),
        sa.Column("model_id", UUID(as_uuid=True), nullable=True),
        sa.Column("cost_usd", sa.Float, nullable=False),
        sa.Column("on_demand_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("efficiency_rate", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("source", sa.String(50), nullable=False, server_default="opencost"),
        sa.Column("raw_metadata", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_fin_cost_records_tenant_period", "fin_cost_records", ["tenant_id", "period_start", "period_end"])
    op.create_index("ix_fin_cost_records_tenant_model", "fin_cost_records", ["tenant_id", "model_id"])
    op.create_index("ix_fin_cost_records_period_start", "fin_cost_records", ["period_start"])
    op.create_index("ix_fin_cost_records_resource_type", "fin_cost_records", ["resource_type"])

    # RLS for fin_cost_records
    op.execute("ALTER TABLE fin_cost_records ENABLE ROW LEVEL SECURITY;")
    op.execute(
        "CREATE POLICY fin_cost_records_tenant_isolation ON fin_cost_records "
        "USING (tenant_id = current_setting('app.current_tenant', true));"
    )

    # fin_token_usage
    op.create_table(
        "fin_token_usage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("model_id", UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_provider", sa.String(100), nullable=False),
        sa.Column("model_tier", sa.String(20), nullable=False, server_default="tier2"),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prompt_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("request_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("prompt_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("completion_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("total_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("workload_name", sa.String(255), nullable=True),
        sa.Column("use_case", sa.String(100), nullable=True),
    )
    op.create_index("ix_fin_token_usage_tenant_model_period", "fin_token_usage", ["tenant_id", "model_id", "period_start"])
    op.create_index("ix_fin_token_usage_tenant_period", "fin_token_usage", ["tenant_id", "period_start"])
    op.create_index("ix_fin_token_usage_model_name", "fin_token_usage", ["model_name"])

    op.execute("ALTER TABLE fin_token_usage ENABLE ROW LEVEL SECURITY;")
    op.execute(
        "CREATE POLICY fin_token_usage_tenant_isolation ON fin_token_usage "
        "USING (tenant_id = current_setting('app.current_tenant', true));"
    )

    # fin_roi_calculations
    op.create_table(
        "fin_roi_calculations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("initiative_name", sa.String(255), nullable=False),
        sa.Column("initiative_type", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("productivity_gain_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("quality_improvement_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("risk_avoidance_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("total_benefit_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("gpu_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("token_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("infra_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("total_ai_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("roi_percent", sa.Float, nullable=False, server_default="0"),
        sa.Column("payback_period_days", sa.Integer, nullable=True),
        sa.Column("assumptions", JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(20), nullable=False, server_default="completed"),
    )
    op.create_index("ix_fin_roi_calculations_tenant_period", "fin_roi_calculations", ["tenant_id", "period_start"])

    op.execute("ALTER TABLE fin_roi_calculations ENABLE ROW LEVEL SECURITY;")
    op.execute(
        "CREATE POLICY fin_roi_calculations_tenant_isolation ON fin_roi_calculations "
        "USING (tenant_id = current_setting('app.current_tenant', true));"
    )

    # fin_budgets
    op.create_table(
        "fin_budgets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("budget_type", sa.String(50), nullable=False, server_default="monthly"),
        sa.Column("scope", sa.String(50), nullable=False, server_default="all"),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("limit_usd", sa.Float, nullable=False),
        sa.Column("warning_threshold", sa.Float, nullable=False, server_default="0.80"),
        sa.Column("critical_threshold", sa.Float, nullable=False, server_default="0.95"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("notification_channels", JSONB, nullable=False, server_default="[]"),
        sa.UniqueConstraint("tenant_id", "name", "period_start", name="uq_fin_budgets_tenant_name_period"),
    )
    op.create_index("ix_fin_budgets_tenant_id", "fin_budgets", ["tenant_id"])
    op.create_index("ix_fin_budgets_name", "fin_budgets", ["name"])

    op.execute("ALTER TABLE fin_budgets ENABLE ROW LEVEL SECURITY;")
    op.execute(
        "CREATE POLICY fin_budgets_tenant_isolation ON fin_budgets "
        "USING (tenant_id = current_setting('app.current_tenant', true));"
    )

    # fin_budget_alerts
    op.create_table(
        "fin_budget_alerts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("budget_id", UUID(as_uuid=True), sa.ForeignKey("fin_budgets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("actual_spend_usd", sa.Float, nullable=False),
        sa.Column("threshold_usd", sa.Float, nullable=False),
        sa.Column("utilization_percent", sa.Float, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("acknowledged", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("acknowledged_by", UUID(as_uuid=True), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("kafka_event_id", sa.String(255), nullable=True),
    )
    op.create_index("ix_fin_budget_alerts_budget_severity", "fin_budget_alerts", ["budget_id", "severity"])
    op.create_index("ix_fin_budget_alerts_tenant_acknowledged", "fin_budget_alerts", ["tenant_id", "acknowledged"])

    op.execute("ALTER TABLE fin_budget_alerts ENABLE ROW LEVEL SECURITY;")
    op.execute(
        "CREATE POLICY fin_budget_alerts_tenant_isolation ON fin_budget_alerts "
        "USING (tenant_id = current_setting('app.current_tenant', true));"
    )

    # fin_routing_recommendations
    op.create_table(
        "fin_routing_recommendations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("workload_name", sa.String(255), nullable=False),
        sa.Column("use_case", sa.String(100), nullable=False),
        sa.Column("quality_requirement", sa.String(20), nullable=False, server_default="standard"),
        sa.Column("latency_requirement_ms", sa.Integer, nullable=True),
        sa.Column("estimated_monthly_requests", sa.Integer, nullable=False, server_default="1000"),
        sa.Column("avg_prompt_tokens", sa.Integer, nullable=False, server_default="500"),
        sa.Column("avg_completion_tokens", sa.Integer, nullable=False, server_default="200"),
        sa.Column("recommended_model_id", UUID(as_uuid=True), nullable=True),
        sa.Column("recommended_model_name", sa.String(255), nullable=False),
        sa.Column("fallback_model_name", sa.String(255), nullable=True),
        sa.Column("routing_score", sa.Float, nullable=False),
        sa.Column("estimated_monthly_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("estimated_savings_vs_premium_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("reasoning", sa.Text, nullable=True),
        sa.Column("candidate_models", JSONB, nullable=False, server_default="[]"),
    )
    op.create_index("ix_fin_routing_tenant_workload", "fin_routing_recommendations", ["tenant_id", "workload_name"])

    op.execute("ALTER TABLE fin_routing_recommendations ENABLE ROW LEVEL SECURITY;")
    op.execute(
        "CREATE POLICY fin_routing_recommendations_tenant_isolation ON fin_routing_recommendations "
        "USING (tenant_id = current_setting('app.current_tenant', true));"
    )


def downgrade() -> None:
    """Drop all fin_ tables."""
    op.drop_table("fin_routing_recommendations")
    op.drop_table("fin_budget_alerts")
    op.drop_table("fin_budgets")
    op.drop_table("fin_roi_calculations")
    op.drop_table("fin_token_usage")
    op.drop_table("fin_cost_records")
