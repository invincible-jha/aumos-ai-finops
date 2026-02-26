"""Service-specific settings extending AumOS base config."""

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Configuration for the AumOS AI FinOps service.

    Extends the standard AumOS settings with FinOps-specific configuration
    for external cost providers, ROI parameters, and GPU pricing.
    """

    service_name: str = "aumos-ai-finops"

    # External cost provider URLs
    opencost_base_url: str = "http://opencost-service:9090"
    kubecost_base_url: str = "http://kubecost-service:9090"
    opencost_enabled: bool = True
    kubecost_enabled: bool = False

    # Cost provider HTTP timeouts (seconds)
    cost_provider_timeout_seconds: int = 30

    # Budget alerting defaults
    default_budget_alert_threshold: float = 0.80  # 80% of budget triggers warning alert
    critical_budget_alert_threshold: float = 0.95  # 95% triggers critical alert

    # ROI calculation defaults
    roi_lookback_days: int = 30
    default_hourly_rate_usd: float = 75.0  # Default fully-loaded engineer hourly rate

    # GPU pricing per hour (USD) — configurable to match cloud provider spot pricing
    gpu_cost_per_hour_a100: float = 2.21  # NVIDIA A100 80GB on-demand
    gpu_cost_per_hour_h100: float = 4.76  # NVIDIA H100 80GB on-demand
    gpu_cost_per_hour_t4: float = 0.35   # NVIDIA T4 for inference

    # Token cost baselines (USD per 1M tokens) for cost-optimized routing
    token_cost_per_million_input_tier1: float = 15.0   # Premium models (GPT-4o, Claude Opus)
    token_cost_per_million_input_tier2: float = 3.0    # Mid-tier models (GPT-4o-mini, Claude Sonnet)
    token_cost_per_million_input_tier3: float = 0.10   # Economy models (GPT-3.5, Claude Haiku)

    # Routing optimization
    routing_cost_weight: float = 0.4       # Weight for cost objective in routing score
    routing_quality_weight: float = 0.4    # Weight for quality objective
    routing_latency_weight: float = 0.2    # Weight for latency objective

    model_config = SettingsConfigDict(env_prefix="AUMOS_FINOPS_")
