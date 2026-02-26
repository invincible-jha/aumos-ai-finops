"""Unit tests for AumOS AI FinOps business logic services."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aumos_ai_finops.core.models import Budget, BudgetAlert, CostRecord, ROICalculation, RoutingRecommendation, TokenUsage
from aumos_ai_finops.core.services import (
    BudgetAlertService,
    CostCollectorService,
    ROIEngineService,
    RoutingOptimizerService,
    TokenTrackerService,
)
from aumos_ai_finops.settings import Settings


# ---------------------------------------------------------------------------
# CostCollectorService tests
# ---------------------------------------------------------------------------


class TestCostCollectorService:
    """Tests for CostCollectorService."""

    @pytest.fixture
    def cost_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.create = AsyncMock()
        repo.list_by_tenant_period = AsyncMock(return_value=[])
        repo.list_gpu_costs = AsyncMock(return_value=[])
        repo.sum_cost_by_tenant_period = AsyncMock(return_value=0.0)
        return repo

    @pytest.fixture
    def service(
        self,
        cost_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        settings: Settings,
    ) -> CostCollectorService:
        return CostCollectorService(
            cost_repo=cost_repo,
            opencost_client=None,
            kubecost_client=None,
            event_publisher=mock_event_publisher,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_record_cost_persists_and_publishes(
        self,
        service: CostCollectorService,
        cost_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        sample_cost_record: CostRecord,
    ) -> None:
        """Recording a cost should persist it and publish a cost_recorded event."""
        cost_repo.create.return_value = sample_cost_record

        result = await service.record_cost(
            tenant_id=tenant_id,
            resource_type="gpu",
            resource_id="gpu-node-001",
            cost_usd=221.0,
            period_start=period_start,
            period_end=period_end,
            gpu_type="a100",
            source="manual",
        )

        assert result.cost_usd == 221.0
        assert result.resource_type == "gpu"
        cost_repo.create.assert_awaited_once()
        mock_event_publisher.publish_cost_recorded.assert_awaited_once_with(
            tenant_id=tenant_id,
            cost_record_id=str(sample_cost_record.id),
            resource_type="gpu",
            cost_usd=221.0,
            period_start=period_start,
            period_end=period_end,
        )

    @pytest.mark.asyncio
    async def test_sync_from_opencost_raises_when_disabled(
        self,
        service: CostCollectorService,
        tenant_id: str,
    ) -> None:
        """Syncing from OpenCost raises RuntimeError when OpenCost is disabled."""
        with pytest.raises(RuntimeError, match="OpenCost integration is not enabled"):
            await service.sync_from_opencost(
                tenant_id=tenant_id,
                namespace=None,
                window="24h",
            )

    @pytest.mark.asyncio
    async def test_get_cost_breakdown_delegates_to_repo(
        self,
        service: CostCollectorService,
        cost_repo: AsyncMock,
        sample_cost_record: CostRecord,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """get_cost_breakdown should return records from the repository."""
        cost_repo.list_by_tenant_period.return_value = [sample_cost_record]

        result = await service.get_cost_breakdown(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )

        assert len(result) == 1
        assert result[0].resource_type == "gpu"
        cost_repo.list_by_tenant_period.assert_awaited_once_with(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            resource_type=None,
        )


# ---------------------------------------------------------------------------
# TokenTrackerService tests
# ---------------------------------------------------------------------------


class TestTokenTrackerService:
    """Tests for TokenTrackerService."""

    @pytest.fixture
    def token_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.create = AsyncMock()
        repo.list_by_tenant_period = AsyncMock(return_value=[])
        repo.aggregate_by_model = AsyncMock(return_value=[])
        repo.sum_cost_by_tenant_period = AsyncMock(return_value=0.0)
        return repo

    @pytest.fixture
    def service(self, token_repo: AsyncMock, settings: Settings) -> TokenTrackerService:
        return TokenTrackerService(token_repo=token_repo, settings=settings)

    @pytest.mark.asyncio
    async def test_record_token_usage_computes_cost(
        self,
        service: TokenTrackerService,
        token_repo: AsyncMock,
        sample_token_usage: TokenUsage,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """Token usage recording should auto-compute costs based on tier pricing."""
        token_repo.create.return_value = sample_token_usage

        result = await service.record_token_usage(
            tenant_id=tenant_id,
            model_id=uuid.uuid4(),
            model_name="claude-sonnet-4",
            model_provider="anthropic",
            model_tier="tier2",
            period_start=period_start,
            period_end=period_end,
            prompt_tokens=100_000,
            completion_tokens=40_000,
        )

        # Cost should be computed (tier2 = $3/M input, $9/M output)
        token_repo.create.assert_awaited_once()
        created_usage: TokenUsage = token_repo.create.call_args[0][0]
        assert created_usage.prompt_cost_usd == pytest.approx(0.30, rel=1e-3)   # 100K * $3/M
        assert created_usage.completion_cost_usd == pytest.approx(0.36, rel=1e-3)  # 40K * $9/M
        assert created_usage.total_cost_usd == pytest.approx(0.66, rel=1e-3)

    def test_compute_token_cost_tier1(self, service: TokenTrackerService) -> None:
        """Tier1 tokens should use premium pricing."""
        prompt_cost, completion_cost = service._compute_token_cost("tier1", 1_000_000, 1_000_000)
        assert prompt_cost == pytest.approx(15.0)        # $15/M input
        assert completion_cost == pytest.approx(45.0)    # $45/M output

    def test_compute_token_cost_tier3(self, service: TokenTrackerService) -> None:
        """Tier3 tokens should use economy pricing."""
        prompt_cost, completion_cost = service._compute_token_cost("tier3", 1_000_000, 1_000_000)
        assert prompt_cost == pytest.approx(0.10)
        assert completion_cost == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# ROIEngineService tests
# ---------------------------------------------------------------------------


class TestROIEngineService:
    """Tests for ROIEngineService."""

    @pytest.fixture
    def roi_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.create = AsyncMock()
        repo.list_by_tenant = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def cost_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.sum_cost_by_tenant_period = AsyncMock(return_value=500.0)
        return repo

    @pytest.fixture
    def token_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.sum_cost_by_tenant_period = AsyncMock(return_value=200.0)
        return repo

    @pytest.fixture
    def service(
        self,
        roi_repo: AsyncMock,
        cost_repo: AsyncMock,
        token_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        settings: Settings,
    ) -> ROIEngineService:
        return ROIEngineService(
            roi_repo=roi_repo,
            cost_repo=cost_repo,
            token_repo=token_repo,
            event_publisher=mock_event_publisher,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_calculate_roi_positive(
        self,
        service: ROIEngineService,
        roi_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """ROI calculation with clear benefit should return positive ROI percent."""
        roi_id = uuid.uuid4()

        def create_side_effect(calc: ROICalculation) -> ROICalculation:
            calc.id = roi_id
            calc.created_at = period_start
            calc.updated_at = period_start
            return calc

        roi_repo.create.side_effect = create_side_effect

        result = await service.calculate_roi(
            tenant_id=tenant_id,
            initiative_name="Code Assistant Pilot",
            initiative_type="code_assistant",
            period_start=period_start,
            period_end=period_end,
            hours_saved=100.0,
            headcount=10,
            hourly_rate_usd=75.0,
            error_reduction_rate=0.2,
            avg_error_cost_usd=500.0,
            incidents_prevented=2,
            avg_incident_cost_usd=1000.0,
        )

        # productivity_gain = 100 * 75 * 10 = 75,000
        # quality_improvement = 0.2 * 500 = 100
        # risk_avoidance = 2 * 1000 = 2,000
        # total_benefit = 77,100
        # total_ai_cost = gpu(500) + token(200) + infra(500-500+0) = 700
        # roi = (77100 - 700) / 700 * 100 ≈ 10,914%
        assert result.roi_percent > 0
        assert result.productivity_gain_usd == pytest.approx(75_000.0)
        assert result.quality_improvement_usd == pytest.approx(100.0)
        assert result.risk_avoidance_usd == pytest.approx(2_000.0)
        mock_event_publisher.publish_roi_calculated.assert_awaited_once()

    def test_compute_payback_period_positive_roi(self, service: ROIEngineService) -> None:
        """Positive net benefit should yield a finite payback period."""
        payback = service._compute_payback_period(
            total_benefit_usd=10_000.0,
            total_ai_cost_usd=1_000.0,
            period_days=30,
        )
        assert payback is not None
        assert payback > 0

    def test_compute_payback_period_negative_roi(self, service: ROIEngineService) -> None:
        """When benefit < cost, payback period should be None."""
        payback = service._compute_payback_period(
            total_benefit_usd=500.0,
            total_ai_cost_usd=1_000.0,
            period_days=30,
        )
        assert payback is None


# ---------------------------------------------------------------------------
# RoutingOptimizerService tests
# ---------------------------------------------------------------------------


class TestRoutingOptimizerService:
    """Tests for RoutingOptimizerService."""

    @pytest.fixture
    def routing_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.create = AsyncMock()
        repo.get_latest_for_workload = AsyncMock(return_value=None)
        return repo

    @pytest.fixture
    def service(self, routing_repo: AsyncMock, settings: Settings) -> RoutingOptimizerService:
        return RoutingOptimizerService(routing_repo=routing_repo, settings=settings)

    @pytest.mark.asyncio
    async def test_optimize_routing_standard_quality(
        self,
        service: RoutingOptimizerService,
        routing_repo: AsyncMock,
        tenant_id: str,
    ) -> None:
        """Standard quality routing should recommend a non-premium model."""
        rec_id = uuid.uuid4()

        def create_side_effect(rec: RoutingRecommendation) -> RoutingRecommendation:
            rec.id = rec_id
            rec.created_at = datetime.now(timezone.utc)
            rec.updated_at = datetime.now(timezone.utc)
            return rec

        routing_repo.create.side_effect = create_side_effect

        result = await service.optimize_routing(
            tenant_id=tenant_id,
            workload_name="code-review-agent",
            use_case="code_generation",
            quality_requirement="standard",
            estimated_monthly_requests=5000,
            avg_prompt_tokens=800,
            avg_completion_tokens=300,
        )

        # Standard quality (min 0.78) should NOT recommend economy models
        assert result.routing_score > 0
        assert result.recommended_model_name is not None
        # Savings should be positive (cheaper than most expensive option)
        assert result.estimated_savings_vs_premium_usd >= 0
        routing_repo.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_optimize_routing_economy_quality(
        self,
        service: RoutingOptimizerService,
        routing_repo: AsyncMock,
        tenant_id: str,
    ) -> None:
        """Economy quality routing should recommend the most cost-effective eligible model."""
        rec_id = uuid.uuid4()

        def create_side_effect(rec: RoutingRecommendation) -> RoutingRecommendation:
            rec.id = rec_id
            rec.created_at = datetime.now(timezone.utc)
            rec.updated_at = datetime.now(timezone.utc)
            return rec

        routing_repo.create.side_effect = create_side_effect

        result = await service.optimize_routing(
            tenant_id=tenant_id,
            workload_name="batch-classifier",
            use_case="classification",
            quality_requirement="economy",
            estimated_monthly_requests=100_000,
        )

        # Economy models should have lower per-request cost
        assert result.estimated_monthly_cost_usd < 100  # Economy should be cheap
        assert len(result.candidate_models) > 0

    def test_compute_token_cost_scales_with_volume(self, service: RoutingOptimizerService) -> None:
        """Token cost should scale linearly with request volume."""
        cost_1k = service._compute_token_cost_usd("tier2", 500, 200, 1_000)
        cost_10k = service._compute_token_cost_usd("tier2", 500, 200, 10_000)
        assert cost_10k == pytest.approx(cost_1k * 10, rel=1e-6)


# ---------------------------------------------------------------------------
# BudgetAlertService tests
# ---------------------------------------------------------------------------


class TestBudgetAlertService:
    """Tests for BudgetAlertService."""

    @pytest.fixture
    def budget_repo(self, sample_budget: Budget) -> AsyncMock:
        repo = AsyncMock()
        repo.create = AsyncMock(return_value=sample_budget)
        repo.list_by_tenant = AsyncMock(return_value=[sample_budget])
        repo.get_active_budgets_for_period = AsyncMock(return_value=[sample_budget])
        return repo

    @pytest.fixture
    def alert_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.create = AsyncMock()
        repo.list_by_budget = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def cost_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.sum_cost_by_tenant_period = AsyncMock(return_value=850.0)  # 85% of $1000 limit
        return repo

    @pytest.fixture
    def token_repo(self) -> AsyncMock:
        repo = AsyncMock()
        repo.sum_cost_by_tenant_period = AsyncMock(return_value=0.0)
        return repo

    @pytest.fixture
    def service(
        self,
        budget_repo: AsyncMock,
        alert_repo: AsyncMock,
        cost_repo: AsyncMock,
        token_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        settings: Settings,
    ) -> BudgetAlertService:
        return BudgetAlertService(
            budget_repo=budget_repo,
            alert_repo=alert_repo,
            cost_repo=cost_repo,
            token_repo=token_repo,
            event_publisher=mock_event_publisher,
            settings=settings,
        )

    @pytest.mark.asyncio
    async def test_evaluate_budgets_triggers_warning_alert(
        self,
        service: BudgetAlertService,
        alert_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        sample_budget: Budget,
        tenant_id: str,
    ) -> None:
        """Spending at 85% of limit should trigger a warning alert."""
        alert_id = uuid.uuid4()

        def create_alert(alert: BudgetAlert) -> BudgetAlert:
            alert.id = alert_id
            alert.created_at = datetime.now(timezone.utc)
            alert.updated_at = datetime.now(timezone.utc)
            return alert

        alert_repo.create.side_effect = create_alert

        alerts = await service.evaluate_budgets(tenant_id=tenant_id)

        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].utilization_percent == pytest.approx(85.0)
        mock_event_publisher.publish_budget_exceeded.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate_budgets_no_alert_under_threshold(
        self,
        service: BudgetAlertService,
        cost_repo: AsyncMock,
        alert_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        tenant_id: str,
    ) -> None:
        """Spending below warning threshold should produce no alerts."""
        cost_repo.sum_cost_by_tenant_period.return_value = 700.0  # 70% — under 80% threshold

        alerts = await service.evaluate_budgets(tenant_id=tenant_id)

        assert len(alerts) == 0
        mock_event_publisher.publish_budget_exceeded.assert_not_awaited()
