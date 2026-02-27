"""Tests for CostOutcomeAttributor.

Covers:
  - DecisionCostComponents.total_cost property
  - record_decision_costs() creates incomplete attribution
  - attribute_outcome() computes ROI correctly
  - attribute_outcome() rejects expired decisions (>90-day window)
  - attribute_outcome() marks attribution_complete=True
  - attribute_outcome() raises when decision not found
  - get_decision_roi() returns attribution for known decisions
  - get_decision_roi() raises when not found
  - _compute_roi() with positive, negative, and zero costs
  - _compute_roi() at break-even
  - WebhookOutcomeAdapter enqueue and pull
  - WebhookOutcomeAdapter tenant isolation
  - WebhookOutcomeAdapter max_events limit
  - WebhookOutcomeAdapter queue_size property
  - ManualEntryOutcomeAdapter submit and pull
  - ManualEntryOutcomeAdapter marks processed after pull
  - ManualEntryOutcomeAdapter tenant isolation
  - OutcomeEvent is a frozen dataclass
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from aumos_ai_finops.attribution.cost_outcome_attributor import (
    BreakEvenAnalysis,
    CostOutcomeAttributor,
    DecisionCostComponents,
    IAttributionRepository,
    OutcomeAttribution,
    UseCaseROISummary,
    _compute_roi,
)
from aumos_ai_finops.attribution.outcome_adapters import IOutcomeAdapter, OutcomeEvent
from aumos_ai_finops.attribution.outcome_adapters.manual import ManualEntryOutcomeAdapter
from aumos_ai_finops.attribution.outcome_adapters.webhook import WebhookOutcomeAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TENANT_ID = "test-tenant-attribution"
AI_SYSTEM_ID = "aumos-text-engine"
USE_CASE = "contract_review"
DECISION_ID = "decision-uuid-001"


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def decision_at(now: datetime) -> datetime:
    return now - timedelta(hours=2)


@pytest.fixture
def incomplete_attribution(decision_at: datetime) -> OutcomeAttribution:
    """Build an incomplete OutcomeAttribution for mock returns."""
    costs = DecisionCostComponents(
        decision_id=DECISION_ID,
        tenant_id=TENANT_ID,
        ai_system_id=AI_SYSTEM_ID,
        use_case=USE_CASE,
        input_token_cost=Decimal("0.015000"),
        output_token_cost=Decimal("0.030000"),
        compute_cost=Decimal("0.005000"),
        storage_cost=Decimal("0.001000"),
        egress_cost=Decimal("0.000500"),
        human_review_cost=Decimal("0.000000"),
        decision_at=decision_at,
    )
    return OutcomeAttribution(
        decision_id=DECISION_ID,
        tenant_id=TENANT_ID,
        ai_system_id=AI_SYSTEM_ID,
        use_case=USE_CASE,
        costs=costs,
        total_cost_usd=costs.total_cost,
        outcome_type=None,
        outcome_value_usd=None,
        roi_pct=None,
        attribution_complete=False,
        decision_at=decision_at,
        outcome_at=None,
    )


@pytest.fixture
def mock_repo() -> AsyncMock:
    """Return a mock IAttributionRepository."""
    repo = AsyncMock(spec=IAttributionRepository)
    repo.save_attribution.return_value = None
    repo.get_by_decision_id.return_value = None
    repo.list_pending_attributions.return_value = []
    repo.get_use_case_summary.return_value = None
    repo.list_attributions_for_dashboard.return_value = []
    return repo


@pytest.fixture
def mock_adapter() -> AsyncMock:
    """Return a mock IOutcomeAdapter."""
    adapter = AsyncMock(spec=IOutcomeAdapter)
    adapter.pull_outcomes.return_value = []
    return adapter


@pytest.fixture
def attributor(mock_repo: AsyncMock, mock_adapter: AsyncMock) -> CostOutcomeAttributor:
    """Return a CostOutcomeAttributor backed by mock repo and adapter."""
    return CostOutcomeAttributor(
        attribution_repo=mock_repo,
        outcome_adapter=mock_adapter,
    )


# ---------------------------------------------------------------------------
# DecisionCostComponents
# ---------------------------------------------------------------------------


def test_decision_cost_components_total() -> None:
    """total_cost property sums all six cost components correctly."""
    costs = DecisionCostComponents(
        decision_id="d-001",
        ai_system_id="sys-1",
        use_case="test",
        decision_at=datetime.now(tz=timezone.utc),
        tenant_id="t1",
        input_token_cost=Decimal("0.010"),
        output_token_cost=Decimal("0.020"),
        compute_cost=Decimal("0.005"),
        storage_cost=Decimal("0.001"),
        egress_cost=Decimal("0.002"),
        human_review_cost=Decimal("0.050"),
    )
    assert costs.total_cost == Decimal("0.088")


def test_decision_cost_components_total_zero() -> None:
    """total_cost is zero when all components are zero."""
    costs = DecisionCostComponents(
        decision_id="d-zero",
        ai_system_id="sys",
        use_case="uc",
        decision_at=datetime.now(tz=timezone.utc),
        tenant_id="t",
        input_token_cost=Decimal("0"),
        output_token_cost=Decimal("0"),
        compute_cost=Decimal("0"),
        storage_cost=Decimal("0"),
        egress_cost=Decimal("0"),
        human_review_cost=Decimal("0"),
    )
    assert costs.total_cost == Decimal("0")


def test_decision_cost_components_is_frozen() -> None:
    """DecisionCostComponents is a frozen dataclass — fields cannot be reassigned."""
    costs = DecisionCostComponents(
        decision_id="d",
        ai_system_id="sys",
        use_case="uc",
        decision_at=datetime.now(tz=timezone.utc),
        tenant_id="t",
        input_token_cost=Decimal("0.01"),
        output_token_cost=Decimal("0"),
        compute_cost=Decimal("0"),
        storage_cost=Decimal("0"),
        egress_cost=Decimal("0"),
        human_review_cost=Decimal("0"),
    )
    with pytest.raises((AttributeError, TypeError)):
        costs.input_token_cost = Decimal("99.99")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# record_decision_costs()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_decision_costs_saves_attribution(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
    decision_at: datetime,
) -> None:
    """record_decision_costs() calls save_attribution once."""
    await attributor.record_decision_costs(
        decision_id=DECISION_ID,
        tenant_id=TENANT_ID,
        ai_system_id=AI_SYSTEM_ID,
        use_case=USE_CASE,
        input_token_cost=Decimal("0.015"),
        output_token_cost=Decimal("0.030"),
        compute_cost=Decimal("0.005"),
        storage_cost=Decimal("0.001"),
        egress_cost=Decimal("0.0005"),
        human_review_cost=Decimal("0"),
        decision_at=decision_at,
    )
    mock_repo.save_attribution.assert_called_once()


@pytest.mark.asyncio
async def test_record_decision_costs_creates_incomplete(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
    decision_at: datetime,
) -> None:
    """record_decision_costs() creates attribution with attribution_complete=False."""
    result = await attributor.record_decision_costs(
        decision_id=DECISION_ID,
        tenant_id=TENANT_ID,
        ai_system_id=AI_SYSTEM_ID,
        use_case=USE_CASE,
        input_token_cost=Decimal("0.015"),
        output_token_cost=Decimal("0.030"),
        compute_cost=Decimal("0.005"),
        storage_cost=Decimal("0.001"),
        egress_cost=Decimal("0.0005"),
        human_review_cost=Decimal("0"),
        decision_at=decision_at,
    )
    assert result.attribution_complete is False
    assert result.outcome_type is None
    assert result.roi_pct is None


@pytest.mark.asyncio
async def test_record_decision_costs_total_cost_matches(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
    decision_at: datetime,
) -> None:
    """record_decision_costs() total_cost_usd is sum of all six components."""
    result = await attributor.record_decision_costs(
        decision_id=DECISION_ID,
        tenant_id=TENANT_ID,
        ai_system_id=AI_SYSTEM_ID,
        use_case=USE_CASE,
        input_token_cost=Decimal("0.010"),
        output_token_cost=Decimal("0.020"),
        compute_cost=Decimal("0.005"),
        storage_cost=Decimal("0.001"),
        egress_cost=Decimal("0.002"),
        human_review_cost=Decimal("0.050"),
        decision_at=decision_at,
    )
    assert result.total_cost_usd == Decimal("0.088")


# ---------------------------------------------------------------------------
# attribute_outcome()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attribute_outcome_computes_positive_roi(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
    incomplete_attribution: OutcomeAttribution,
    now: datetime,
) -> None:
    """attribute_outcome() computes positive ROI when outcome > cost."""
    mock_repo.get_by_decision_id.return_value = incomplete_attribution
    result = await attributor.attribute_outcome(
        decision_id=DECISION_ID,
        tenant_id=TENANT_ID,
        outcome_type="cost_saved",
        outcome_value_usd=Decimal("5.00"),
        outcome_at=now,
    )
    assert result.attribution_complete is True
    assert result.outcome_value_usd == Decimal("5.00")
    assert result.roi_pct is not None
    assert result.roi_pct > Decimal("0")


@pytest.mark.asyncio
async def test_attribute_outcome_marks_complete(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
    incomplete_attribution: OutcomeAttribution,
    now: datetime,
) -> None:
    """attribute_outcome() sets attribution_complete=True after successful match."""
    mock_repo.get_by_decision_id.return_value = incomplete_attribution
    result = await attributor.attribute_outcome(
        decision_id=DECISION_ID,
        tenant_id=TENANT_ID,
        outcome_type="revenue_generated",
        outcome_value_usd=Decimal("100.00"),
        outcome_at=now,
    )
    assert result.attribution_complete is True
    mock_repo.save_attribution.assert_called()


@pytest.mark.asyncio
async def test_attribute_outcome_rejects_expired_window(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
) -> None:
    """attribute_outcome() raises ValueError when decision is > attribution_window_days old."""
    old_decision_at = datetime(2025, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
    costs = DecisionCostComponents(
        decision_id="d-expired",
        ai_system_id=AI_SYSTEM_ID,
        use_case=USE_CASE,
        decision_at=old_decision_at,
        tenant_id=TENANT_ID,
        input_token_cost=Decimal("0.05"),
        output_token_cost=Decimal("0"),
        compute_cost=Decimal("0"),
        storage_cost=Decimal("0"),
        egress_cost=Decimal("0"),
        human_review_cost=Decimal("0"),
    )
    expired = OutcomeAttribution(
        decision_id="d-expired",
        tenant_id=TENANT_ID,
        ai_system_id=AI_SYSTEM_ID,
        use_case=USE_CASE,
        costs=costs,
        total_cost_usd=Decimal("0.05"),
        outcome_type=None,
        outcome_value_usd=None,
        roi_pct=None,
        attribution_complete=False,
        decision_at=old_decision_at,
        outcome_at=None,
    )
    mock_repo.get_by_decision_id.return_value = expired

    with pytest.raises(ValueError, match="90"):
        await attributor.attribute_outcome(
            decision_id="d-expired",
            tenant_id=TENANT_ID,
            outcome_type="cost_saved",
            outcome_value_usd=Decimal("1.00"),
            outcome_at=datetime(2026, 2, 26, 0, 0, 0, tzinfo=timezone.utc),
        )


@pytest.mark.asyncio
async def test_attribute_outcome_not_found_raises(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
    now: datetime,
) -> None:
    """attribute_outcome() raises ValueError when no decision record exists."""
    mock_repo.get_by_decision_id.return_value = None
    with pytest.raises(ValueError, match="not found|No cost record"):
        await attributor.attribute_outcome(
            decision_id="nonexistent-decision",
            tenant_id=TENANT_ID,
            outcome_type="cost_saved",
            outcome_value_usd=Decimal("1.00"),
            outcome_at=now,
        )


# ---------------------------------------------------------------------------
# get_decision_roi()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_decision_roi_returns_attribution(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
    incomplete_attribution: OutcomeAttribution,
) -> None:
    """get_decision_roi() returns the attribution for a known decision."""
    mock_repo.get_by_decision_id.return_value = incomplete_attribution
    result = await attributor.get_decision_roi(DECISION_ID, TENANT_ID)
    assert result.decision_id == DECISION_ID
    assert result.tenant_id == TENANT_ID


@pytest.mark.asyncio
async def test_get_decision_roi_raises_when_not_found(
    attributor: CostOutcomeAttributor,
    mock_repo: AsyncMock,
) -> None:
    """get_decision_roi() raises ValueError when decision is unknown."""
    mock_repo.get_by_decision_id.return_value = None
    with pytest.raises(ValueError, match="No attribution record"):
        await attributor.get_decision_roi("unknown-decision", TENANT_ID)


# ---------------------------------------------------------------------------
# _compute_roi() helper
# ---------------------------------------------------------------------------


def test_compute_roi_positive() -> None:
    """ROI is positive when outcome > cost."""
    roi = _compute_roi(Decimal("10.00"), Decimal("100.00"))
    assert roi == pytest.approx(Decimal("900.00"), abs=Decimal("0.01"))


def test_compute_roi_negative() -> None:
    """ROI is negative when outcome < cost (loss-making decision)."""
    roi = _compute_roi(Decimal("10.00"), Decimal("5.00"))
    assert roi < Decimal("0")


def test_compute_roi_zero_cost_returns_zero() -> None:
    """_compute_roi returns Decimal('0') when total_cost is zero."""
    roi = _compute_roi(Decimal("0"), Decimal("100.00"))
    assert roi == Decimal("0")


def test_compute_roi_at_breakeven() -> None:
    """ROI is 0% when outcome equals cost."""
    roi = _compute_roi(Decimal("50.00"), Decimal("50.00"))
    assert roi == pytest.approx(Decimal("0"), abs=Decimal("0.001"))


def test_compute_roi_uses_decimal_not_float() -> None:
    """_compute_roi accepts and returns Decimal, not float."""
    roi = _compute_roi(Decimal("0.0515"), Decimal("5.00"))
    assert isinstance(roi, Decimal)


# ---------------------------------------------------------------------------
# WebhookOutcomeAdapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_adapter_enqueue_increases_size() -> None:
    """Enqueuing an event increases queue_size by 1."""
    adapter = WebhookOutcomeAdapter()
    assert adapter.queue_size == 0
    await adapter.enqueue_outcome(
        decision_id="d-001",
        tenant_id="t1",
        outcome_type="cost_saved",
        outcome_value_usd=Decimal("1.00"),
    )
    assert adapter.queue_size == 1


@pytest.mark.asyncio
async def test_webhook_adapter_pull_drains_correct_tenant() -> None:
    """pull_outcomes returns only events for the specified tenant."""
    adapter = WebhookOutcomeAdapter()
    await adapter.enqueue_outcome("d-a1", "tenant-a", "cost_saved", Decimal("1.00"))
    await adapter.enqueue_outcome("d-b1", "tenant-b", "revenue_generated", Decimal("2.00"))
    await adapter.enqueue_outcome("d-a2", "tenant-a", "time_saved", Decimal("3.00"))

    pulled = await adapter.pull_outcomes(tenant_id="tenant-a", limit=10)
    assert len(pulled) == 2
    assert all(e.tenant_id == "tenant-a" for e in pulled)


@pytest.mark.asyncio
async def test_webhook_adapter_pull_respects_limit() -> None:
    """pull_outcomes returns at most `limit` events."""
    adapter = WebhookOutcomeAdapter()
    for i in range(5):
        await adapter.enqueue_outcome(f"d-{i}", "tenant-lim", "cost_saved", Decimal("1.00"))

    pulled = await adapter.pull_outcomes(tenant_id="tenant-lim", limit=3)
    assert len(pulled) == 3


@pytest.mark.asyncio
async def test_webhook_adapter_pull_empties_tenant_queue() -> None:
    """After a full pull, queue size drops by the number of tenant events."""
    adapter = WebhookOutcomeAdapter()
    await adapter.enqueue_outcome("d-1", "tenant-drain", "cost_saved", Decimal("1.00"))
    await adapter.enqueue_outcome("d-2", "tenant-other", "cost_saved", Decimal("1.00"))
    await adapter.pull_outcomes(tenant_id="tenant-drain", limit=10)
    # Only the "tenant-other" event should remain
    assert adapter.queue_size == 1


@pytest.mark.asyncio
async def test_webhook_adapter_events_have_webhook_source() -> None:
    """Enqueued events have source='webhook'."""
    adapter = WebhookOutcomeAdapter()
    await adapter.enqueue_outcome("d-src", "t", "cost_saved", Decimal("1.00"))
    pulled = await adapter.pull_outcomes(tenant_id="t", limit=1)
    assert pulled[0].source == "webhook"


# ---------------------------------------------------------------------------
# ManualEntryOutcomeAdapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_adapter_submit_and_pull() -> None:
    """submit_outcome persists event; pull_outcomes returns it."""
    adapter = ManualEntryOutcomeAdapter()
    await adapter.submit_outcome(
        decision_id="d-manual-001",
        tenant_id="tenant-manual",
        outcome_type="time_saved",
        outcome_value_usd=Decimal("75.00"),
    )
    pulled = await adapter.pull_outcomes(tenant_id="tenant-manual", limit=10)
    assert len(pulled) == 1
    assert pulled[0].decision_id == "d-manual-001"


@pytest.mark.asyncio
async def test_manual_adapter_marks_processed_after_pull() -> None:
    """Events are not returned on a second pull after being processed."""
    adapter = ManualEntryOutcomeAdapter()
    await adapter.submit_outcome("d-once", "t-once", "cost_saved", Decimal("10.00"))

    first = await adapter.pull_outcomes(tenant_id="t-once", limit=10)
    assert len(first) == 1

    second = await adapter.pull_outcomes(tenant_id="t-once", limit=10)
    assert len(second) == 0


@pytest.mark.asyncio
async def test_manual_adapter_tenant_isolation() -> None:
    """pull_outcomes only returns events for the specified tenant."""
    adapter = ManualEntryOutcomeAdapter()
    await adapter.submit_outcome("d-x", "tenant-x", "cost_saved", Decimal("100.00"))
    await adapter.submit_outcome("d-y", "tenant-y", "revenue_generated", Decimal("200.00"))

    pulled_x = await adapter.pull_outcomes(tenant_id="tenant-x", limit=10)
    assert len(pulled_x) == 1
    assert pulled_x[0].tenant_id == "tenant-x"


@pytest.mark.asyncio
async def test_manual_adapter_returns_event_with_manual_source() -> None:
    """Submitted events have source='manual'."""
    adapter = ManualEntryOutcomeAdapter()
    event = await adapter.submit_outcome("d-src", "t", "cost_saved", Decimal("1.00"))
    assert event.source == "manual"


# ---------------------------------------------------------------------------
# OutcomeEvent frozen
# ---------------------------------------------------------------------------


def test_outcome_event_is_frozen() -> None:
    """OutcomeEvent is a frozen dataclass and cannot be mutated."""
    event = OutcomeEvent(
        decision_id="d-hash",
        tenant_id="t",
        outcome_type="cost_saved",
        outcome_value_usd=Decimal("1.00"),
        outcome_at=datetime.now(tz=timezone.utc),
        source="webhook",
    )
    with pytest.raises((AttributeError, TypeError)):
        event.outcome_type = "revenue_generated"  # type: ignore[misc]
