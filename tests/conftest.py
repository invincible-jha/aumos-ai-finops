"""Shared test fixtures for aumos-ai-finops tests."""

import sys
from pathlib import Path

# Ensure the src/ layout is importable without installing the package.
_SRC_PATH = Path(__file__).parent.parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from aumos_ai_finops.core.models import Budget, CostRecord, ROICalculation, TokenUsage
from aumos_ai_finops.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Provide test settings with safe defaults."""
    return Settings(
        opencost_enabled=False,
        kubecost_enabled=False,
        opencost_base_url="http://opencost-test:9090",
        kubecost_base_url="http://kubecost-test:9090",
        default_budget_alert_threshold=0.80,
        critical_budget_alert_threshold=0.95,
        roi_lookback_days=30,
        default_hourly_rate_usd=75.0,
        gpu_cost_per_hour_a100=2.21,
        token_cost_per_million_input_tier1=15.0,
        token_cost_per_million_input_tier2=3.0,
        token_cost_per_million_input_tier3=0.10,
    )


@pytest.fixture
def tenant_id() -> str:
    """Provide a consistent test tenant ID."""
    return "test-tenant-001"


@pytest.fixture
def now() -> datetime:
    """Provide a consistent reference time."""
    return datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def period_start(now: datetime) -> datetime:
    """Start of the test period (30 days ago)."""
    return now - timedelta(days=30)


@pytest.fixture
def period_end(now: datetime) -> datetime:
    """End of the test period (now)."""
    return now


@pytest.fixture
def mock_event_publisher() -> AsyncMock:
    """Mock FinOps event publisher — all publish methods are async no-ops."""
    publisher = AsyncMock()
    publisher.publish_cost_recorded = AsyncMock()
    publisher.publish_budget_exceeded = AsyncMock()
    publisher.publish_roi_calculated = AsyncMock()
    return publisher


@pytest.fixture
def sample_cost_record(tenant_id: str, period_start: datetime, period_end: datetime) -> CostRecord:
    """Build a sample CostRecord for testing."""
    record = CostRecord(
        tenant_id=tenant_id,
        resource_type="gpu",
        resource_id="gpu-node-001",
        gpu_type="a100",
        cost_usd=221.0,
        on_demand_cost_usd=221.0,
        efficiency_rate=0.85,
        period_start=period_start,
        period_end=period_end,
        source="manual",
        raw_metadata={},
    )
    record.id = uuid.uuid4()
    record.created_at = period_start
    record.updated_at = period_start
    return record


@pytest.fixture
def sample_token_usage(tenant_id: str, period_start: datetime, period_end: datetime) -> TokenUsage:
    """Build a sample TokenUsage record for testing."""
    usage = TokenUsage(
        tenant_id=tenant_id,
        model_id=uuid.uuid4(),
        model_name="claude-sonnet-4",
        model_provider="anthropic",
        model_tier="tier2",
        period_start=period_start,
        period_end=period_end,
        prompt_tokens=100_000,
        completion_tokens=40_000,
        total_tokens=140_000,
        request_count=200,
        prompt_cost_usd=0.30,
        completion_cost_usd=0.36,
        total_cost_usd=0.66,
    )
    usage.id = uuid.uuid4()
    usage.created_at = period_start
    usage.updated_at = period_start
    return usage


@pytest.fixture
def sample_budget(tenant_id: str, period_start: datetime, period_end: datetime) -> Budget:
    """Build a sample Budget for testing."""
    budget = Budget(
        tenant_id=tenant_id,
        name="Test Monthly Budget",
        budget_type="monthly",
        scope="all",
        period_start=period_start,
        period_end=period_end,
        limit_usd=1000.0,
        warning_threshold=0.80,
        critical_threshold=0.95,
        is_active=True,
        notification_channels=[],
    )
    budget.id = uuid.uuid4()
    budget.created_at = period_start
    budget.updated_at = period_start
    return budget
