"""Unit tests for FinOpsService.

Verifies:
  - get_cost_breakdown() aggregates by team, project, model, service
  - create_budget() persists and returns a BudgetLimitResponse
  - get_budget_status() computes consumed, remaining, utilisation, projection
  - get_budget_status() raises RuntimeError when no budget found
  - generate_chargeback_report() produces correct totals and line items
  - generate_chargeback_report() produces CSV when format=csv
  - Budget alert logic fires when utilisation >= alert_threshold_pct
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_ai_finops.api.schemas.finops import (
    BudgetLimitResponse,
    BudgetStatusResponse,
    ChargebackReportResponse,
    CostBreakdownResponse,
)
from aumos_ai_finops.core.models.cost_allocation import BudgetLimit, CostAllocation
from aumos_ai_finops.core.services.finops_service import FinOpsService
from aumos_ai_finops.settings import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(**kwargs) -> Settings:
    """Build Settings with test-safe environment."""
    return Settings(
        service_name="test-finops",
        database_url="postgresql+asyncpg://test:test@localhost/test",
        enable_hard_budget_caps=kwargs.get("enable_hard_budget_caps", False),
    )


def _make_allocation(
    tenant_id: str = "t-1",
    team_id: str = "team-a",
    project_id: str = "proj-1",
    service: str = "aumos-llm-serving",
    model_id: str = "gpt-4o",
    total_cost_usd: float = 100.0,
    total_input_tokens: int = 10_000,
    total_output_tokens: int = 5_000,
    inference_minutes: float = 2.5,
    storage_gb_days: float = 0.5,
) -> CostAllocation:
    """Build a CostAllocation test fixture."""
    now = datetime.now(timezone.utc)
    alloc = CostAllocation(
        tenant_id=tenant_id,
        team_id=team_id,
        project_id=project_id,
        service=service,
        model_id=model_id,
        total_cost_usd=total_cost_usd,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        inference_minutes=inference_minutes,
        storage_gb_days=storage_gb_days,
        period_start=now - timedelta(days=30),
        period_end=now,
    )
    alloc.id = uuid.uuid4()
    return alloc


def _make_budget_limit(
    tenant_id: str = "t-1",
    team_id: str | None = "team-a",
    period_type: str = "monthly",
    limit_usd: float = 1000.0,
    alert_threshold_pct: int = 80,
    hard_cap: bool = False,
) -> BudgetLimit:
    """Build a BudgetLimit test fixture."""
    limit = BudgetLimit(
        tenant_id=tenant_id,
        team_id=team_id,
        period_type=period_type,
        limit_usd=limit_usd,
        alert_threshold_pct=alert_threshold_pct,
        hard_cap=hard_cap,
        is_active=True,
    )
    limit.id = uuid.uuid4()
    limit.created_at = datetime.now(timezone.utc)
    limit.updated_at = datetime.now(timezone.utc)
    return limit


def _make_service(
    allocations: list[CostAllocation] | None = None,
    budget_limit: BudgetLimit | None = None,
    current_cost_usd: float = 0.0,
    current_token_usd: float = 0.0,
    settings: Settings | None = None,
) -> FinOpsService:
    """Build a FinOpsService with mocked repositories."""
    allocation_repo = MagicMock()
    allocation_repo.list_by_tenant_period = AsyncMock(return_value=allocations or [])
    allocation_repo.list_all_for_report = AsyncMock(return_value=allocations or [])

    budget_limit_repo = MagicMock()
    budget_limit_repo.create = AsyncMock(side_effect=lambda limit: limit)
    budget_limit_repo.get_by_team = AsyncMock(return_value=budget_limit)

    cost_repo = MagicMock()
    cost_repo.sum_cost_by_tenant_period = AsyncMock(return_value=current_cost_usd)

    token_repo = MagicMock()
    token_repo.sum_cost_by_tenant_period = AsyncMock(return_value=current_token_usd)

    return FinOpsService(
        allocation_repo=allocation_repo,
        budget_limit_repo=budget_limit_repo,
        cost_repo=cost_repo,
        token_repo=token_repo,
        settings=settings or _make_settings(),
    )


# ---------------------------------------------------------------------------
# get_cost_breakdown() tests
# ---------------------------------------------------------------------------


class TestGetCostBreakdown:
    """Tests for FinOpsService.get_cost_breakdown()."""

    @pytest.mark.asyncio
    async def test_returns_cost_breakdown_response(self) -> None:
        svc = _make_service(allocations=[_make_allocation()])
        now = datetime.now(timezone.utc)
        result = await svc.get_cost_breakdown(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert isinstance(result, CostBreakdownResponse)

    @pytest.mark.asyncio
    async def test_total_cost_sums_all_allocations(self) -> None:
        allocations = [
            _make_allocation(team_id="a", total_cost_usd=100.0),
            _make_allocation(team_id="b", total_cost_usd=200.0),
        ]
        svc = _make_service(allocations=allocations)
        now = datetime.now(timezone.utc)
        result = await svc.get_cost_breakdown(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.total_cost_usd == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_group_by_team_creates_team_line_items(self) -> None:
        allocations = [
            _make_allocation(team_id="team-a", total_cost_usd=50.0),
            _make_allocation(team_id="team-a", total_cost_usd=50.0),  # Same team, should merge
            _make_allocation(team_id="team-b", total_cost_usd=100.0),
        ]
        svc = _make_service(allocations=allocations)
        now = datetime.now(timezone.utc)
        result = await svc.get_cost_breakdown(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
            group_by="team",
        )
        team_keys = {item.group_key for item in result.line_items}
        assert "team-a" in team_keys
        assert "team-b" in team_keys
        team_a_item = next(i for i in result.line_items if i.group_key == "team-a")
        assert team_a_item.total_cost_usd == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_group_by_model_creates_model_line_items(self) -> None:
        allocations = [
            _make_allocation(model_id="gpt-4o", total_cost_usd=80.0),
            _make_allocation(model_id="claude-opus-4", total_cost_usd=120.0),
        ]
        svc = _make_service(allocations=allocations)
        now = datetime.now(timezone.utc)
        result = await svc.get_cost_breakdown(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
            group_by="model",
        )
        assert result.group_by == "model"
        model_keys = {item.group_key for item in result.line_items}
        assert "gpt-4o" in model_keys

    @pytest.mark.asyncio
    async def test_line_items_sorted_by_cost_descending(self) -> None:
        allocations = [
            _make_allocation(team_id="cheap-team", total_cost_usd=10.0),
            _make_allocation(team_id="expensive-team", total_cost_usd=500.0),
        ]
        svc = _make_service(allocations=allocations)
        now = datetime.now(timezone.utc)
        result = await svc.get_cost_breakdown(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        costs = [item.total_cost_usd for item in result.line_items]
        assert costs == sorted(costs, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_allocations_returns_zero_total(self) -> None:
        svc = _make_service(allocations=[])
        now = datetime.now(timezone.utc)
        result = await svc.get_cost_breakdown(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.total_cost_usd == 0.0
        assert result.line_items == []


# ---------------------------------------------------------------------------
# create_budget() tests
# ---------------------------------------------------------------------------


class TestCreateBudget:
    """Tests for FinOpsService.create_budget()."""

    @pytest.mark.asyncio
    async def test_returns_budget_limit_response(self) -> None:
        svc = _make_service()
        result = await svc.create_budget(
            tenant_id="t-1",
            period_type="monthly",
            limit_usd=5000.0,
        )
        assert isinstance(result, BudgetLimitResponse)

    @pytest.mark.asyncio
    async def test_hard_cap_disabled_by_default(self) -> None:
        svc = _make_service(settings=_make_settings(enable_hard_budget_caps=False))
        result = await svc.create_budget(
            tenant_id="t-1",
            period_type="monthly",
            limit_usd=1000.0,
            hard_cap=True,  # Requested but service setting overrides
        )
        assert result.hard_cap is False  # Setting=False means hard_cap is disabled

    @pytest.mark.asyncio
    async def test_hard_cap_enabled_when_setting_allows(self) -> None:
        svc = _make_service(settings=_make_settings(enable_hard_budget_caps=True))
        result = await svc.create_budget(
            tenant_id="t-1",
            period_type="monthly",
            limit_usd=1000.0,
            hard_cap=True,
        )
        assert result.hard_cap is True

    @pytest.mark.asyncio
    async def test_team_id_stored_correctly(self) -> None:
        svc = _make_service()
        result = await svc.create_budget(
            tenant_id="t-1",
            period_type="quarterly",
            limit_usd=10_000.0,
            team_id="team-x",
        )
        assert result.team_id == "team-x"


# ---------------------------------------------------------------------------
# get_budget_status() tests
# ---------------------------------------------------------------------------


class TestGetBudgetStatus:
    """Tests for FinOpsService.get_budget_status()."""

    @pytest.mark.asyncio
    async def test_raises_when_no_budget_limit_found(self) -> None:
        svc = _make_service(budget_limit=None)
        now = datetime.now(timezone.utc)
        with pytest.raises(RuntimeError, match="No active"):
            await svc.get_budget_status(
                tenant_id="t-1",
                team_id="team-z",
                period_start=now - timedelta(days=30),
                period_end=now,
            )

    @pytest.mark.asyncio
    async def test_consumed_combines_cost_and_token_spend(self) -> None:
        budget = _make_budget_limit(limit_usd=1000.0)
        svc = _make_service(
            budget_limit=budget,
            current_cost_usd=200.0,
            current_token_usd=150.0,
        )
        now = datetime.now(timezone.utc)
        result = await svc.get_budget_status(
            tenant_id="t-1",
            team_id="team-a",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.consumed_usd == pytest.approx(350.0)

    @pytest.mark.asyncio
    async def test_remaining_calculated_correctly(self) -> None:
        budget = _make_budget_limit(limit_usd=1000.0)
        svc = _make_service(budget_limit=budget, current_cost_usd=300.0, current_token_usd=100.0)
        now = datetime.now(timezone.utc)
        result = await svc.get_budget_status(
            tenant_id="t-1",
            team_id="team-a",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.remaining_usd == pytest.approx(600.0)

    @pytest.mark.asyncio
    async def test_utilisation_pct_calculated(self) -> None:
        budget = _make_budget_limit(limit_usd=1000.0)
        svc = _make_service(budget_limit=budget, current_cost_usd=500.0, current_token_usd=0.0)
        now = datetime.now(timezone.utc)
        result = await svc.get_budget_status(
            tenant_id="t-1",
            team_id="team-a",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.utilization_pct == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_alert_triggered_when_at_threshold(self) -> None:
        budget = _make_budget_limit(limit_usd=1000.0, alert_threshold_pct=80)
        svc = _make_service(budget_limit=budget, current_cost_usd=850.0, current_token_usd=0.0)
        now = datetime.now(timezone.utc)
        result = await svc.get_budget_status(
            tenant_id="t-1",
            team_id="team-a",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.alert_triggered is True

    @pytest.mark.asyncio
    async def test_alert_not_triggered_below_threshold(self) -> None:
        budget = _make_budget_limit(limit_usd=1000.0, alert_threshold_pct=80)
        svc = _make_service(budget_limit=budget, current_cost_usd=500.0, current_token_usd=0.0)
        now = datetime.now(timezone.utc)
        result = await svc.get_budget_status(
            tenant_id="t-1",
            team_id="team-a",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.alert_triggered is False

    @pytest.mark.asyncio
    async def test_returns_budget_status_response_type(self) -> None:
        budget = _make_budget_limit()
        svc = _make_service(budget_limit=budget)
        now = datetime.now(timezone.utc)
        result = await svc.get_budget_status(
            tenant_id="t-1",
            team_id="team-a",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert isinstance(result, BudgetStatusResponse)


# ---------------------------------------------------------------------------
# generate_chargeback_report() tests
# ---------------------------------------------------------------------------


class TestGenerateChargebackReport:
    """Tests for FinOpsService.generate_chargeback_report()."""

    @pytest.mark.asyncio
    async def test_returns_chargeback_report_response(self) -> None:
        svc = _make_service(allocations=[_make_allocation()])
        now = datetime.now(timezone.utc)
        result = await svc.generate_chargeback_report(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert isinstance(result, ChargebackReportResponse)

    @pytest.mark.asyncio
    async def test_total_cost_matches_sum_of_allocations(self) -> None:
        allocations = [
            _make_allocation(total_cost_usd=100.0),
            _make_allocation(team_id="team-b", total_cost_usd=250.0),
        ]
        svc = _make_service(allocations=allocations)
        now = datetime.now(timezone.utc)
        result = await svc.generate_chargeback_report(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.total_cost_usd == pytest.approx(350.0)

    @pytest.mark.asyncio
    async def test_line_items_sorted_by_cost_descending(self) -> None:
        allocations = [
            _make_allocation(team_id="a", total_cost_usd=50.0),
            _make_allocation(team_id="b", total_cost_usd=300.0),
            _make_allocation(team_id="c", total_cost_usd=150.0),
        ]
        svc = _make_service(allocations=allocations)
        now = datetime.now(timezone.utc)
        result = await svc.generate_chargeback_report(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        costs = [item.total_cost_usd for item in result.line_items]
        assert costs == sorted(costs, reverse=True)

    @pytest.mark.asyncio
    async def test_csv_format_produces_csv_data(self) -> None:
        svc = _make_service(allocations=[_make_allocation()])
        now = datetime.now(timezone.utc)
        result = await svc.generate_chargeback_report(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
            report_format="csv",
        )
        assert result.format == "csv"
        assert result.csv_data is not None
        assert "team_id" in result.csv_data  # CSV header row

    @pytest.mark.asyncio
    async def test_json_format_has_no_csv_data(self) -> None:
        svc = _make_service(allocations=[_make_allocation()])
        now = datetime.now(timezone.utc)
        result = await svc.generate_chargeback_report(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
            report_format="json",
        )
        assert result.format == "json"
        assert result.csv_data is None

    @pytest.mark.asyncio
    async def test_empty_report_has_zero_total(self) -> None:
        svc = _make_service(allocations=[])
        now = datetime.now(timezone.utc)
        result = await svc.generate_chargeback_report(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        assert result.total_cost_usd == 0.0
        assert result.line_items == []

    @pytest.mark.asyncio
    async def test_generated_at_is_recent(self) -> None:
        svc = _make_service(allocations=[])
        now = datetime.now(timezone.utc)
        result = await svc.generate_chargeback_report(
            tenant_id="t-1",
            period_start=now - timedelta(days=30),
            period_end=now,
        )
        age_seconds = (datetime.now(timezone.utc) - result.generated_at).total_seconds()
        assert age_seconds < 5  # Generated within the last 5 seconds


class TestGenerateCsv:
    """Tests for the static CSV generation helper."""

    def test_csv_contains_header(self) -> None:
        from aumos_ai_finops.api.schemas.finops import ChargebackLineItem
        from aumos_ai_finops.core.services.finops_service import FinOpsService
        items = [
            ChargebackLineItem(
                team_id="t",
                project_id="p",
                service="svc",
                model_id="m",
                total_input_tokens=100,
                total_output_tokens=50,
                total_cost_usd=5.0,
                inference_minutes=1.0,
                storage_gb_days=0.1,
            )
        ]
        csv_str = FinOpsService._generate_csv(items)
        assert "team_id" in csv_str
        assert "total_cost_usd" in csv_str

    def test_csv_contains_data_row(self) -> None:
        from aumos_ai_finops.api.schemas.finops import ChargebackLineItem
        from aumos_ai_finops.core.services.finops_service import FinOpsService
        items = [
            ChargebackLineItem(
                team_id="my-team",
                project_id="proj-99",
                service="llm",
                model_id="gpt-4o",
                total_input_tokens=1000,
                total_output_tokens=500,
                total_cost_usd=12.34,
                inference_minutes=5.0,
                storage_gb_days=0.5,
            )
        ]
        csv_str = FinOpsService._generate_csv(items)
        assert "my-team" in csv_str
        assert "12.34" in csv_str
