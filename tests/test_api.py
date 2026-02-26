"""API endpoint tests for AumOS AI FinOps.

Tests validate request/response shapes and service delegation.
Full integration tests require testcontainers (postgres + kafka).
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from aumos_ai_finops.core.models import Budget, CostRecord, ROICalculation, RoutingRecommendation, TokenUsage
from aumos_ai_finops.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tenant() -> dict[str, str]:
    """Simulate an authenticated tenant context."""
    return {"tenant_id": "test-tenant-001", "user_id": str(uuid.uuid4())}


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for test requests."""
    return {"Authorization": "Bearer test-token", "X-Tenant-ID": "test-tenant-001"}


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 2, 26, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def period_start(now: datetime) -> datetime:
    return now - timedelta(days=30)


@pytest.fixture
def period_end(now: datetime) -> datetime:
    return now


@pytest.fixture
def sample_cost_record_data(period_start: datetime, period_end: datetime) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "tenant_id": "test-tenant-001",
        "resource_type": "gpu",
        "resource_id": "gpu-node-001",
        "gpu_type": "a100",
        "workload_name": "llm-inference",
        "namespace": "ai-prod",
        "model_id": None,
        "cost_usd": 221.0,
        "on_demand_cost_usd": 221.0,
        "efficiency_rate": 0.85,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "source": "manual",
        "created_at": period_start.isoformat(),
    }


# ---------------------------------------------------------------------------
# Cost endpoint tests
# ---------------------------------------------------------------------------


class TestCostEndpoints:
    """Tests for /api/v1/finops/costs endpoints."""

    def test_record_cost_request_validates_resource_type(
        self,
        auth_headers: dict[str, str],
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """Invalid resource_type should return 422 validation error."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/finops/costs/record",
                headers=auth_headers,
                json={
                    "resource_type": "invalid_type",  # Not in allowed values
                    "resource_id": "test",
                    "cost_usd": 100.0,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                },
            )
        assert response.status_code == 422

    def test_record_cost_requires_positive_cost(
        self,
        auth_headers: dict[str, str],
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """Zero or negative cost_usd should return 422."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/finops/costs/record",
                headers=auth_headers,
                json={
                    "resource_type": "gpu",
                    "resource_id": "test",
                    "cost_usd": -10.0,  # Invalid
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                },
            )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# ROI endpoint tests
# ---------------------------------------------------------------------------


class TestROIEndpoints:
    """Tests for /api/v1/finops/roi endpoints."""

    def test_roi_calculate_requires_positive_headcount(
        self,
        auth_headers: dict[str, str],
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """headcount < 1 should return 422."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/finops/roi/calculate",
                headers=auth_headers,
                json={
                    "initiative_name": "Test Initiative",
                    "initiative_type": "code_assistant",
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "hours_saved": 100.0,
                    "headcount": 0,  # Invalid — must be >= 1
                },
            )
        assert response.status_code == 422

    def test_roi_calculate_validates_initiative_name_min_length(
        self,
        auth_headers: dict[str, str],
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """Initiative name shorter than 3 chars should return 422."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/finops/roi/calculate",
                headers=auth_headers,
                json={
                    "initiative_name": "AB",  # Too short
                    "initiative_type": "code_assistant",
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "hours_saved": 100.0,
                    "headcount": 5,
                },
            )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Budget endpoint tests
# ---------------------------------------------------------------------------


class TestBudgetEndpoints:
    """Tests for /api/v1/finops/budgets endpoints."""

    def test_create_budget_requires_positive_limit(
        self,
        auth_headers: dict[str, str],
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """limit_usd <= 0 should return 422."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/finops/budgets",
                headers=auth_headers,
                json={
                    "name": "Test Budget",
                    "limit_usd": 0,  # Invalid
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                },
            )
        assert response.status_code == 422

    def test_create_budget_validates_budget_type(
        self,
        auth_headers: dict[str, str],
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """Invalid budget_type should return 422."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/finops/budgets",
                headers=auth_headers,
                json={
                    "name": "Test Budget",
                    "limit_usd": 1000.0,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "budget_type": "biweekly",  # Not allowed
                },
            )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Routing endpoint tests
# ---------------------------------------------------------------------------


class TestRoutingEndpoints:
    """Tests for /api/v1/finops/routing endpoints."""

    def test_routing_request_validates_quality_requirement(
        self,
        auth_headers: dict[str, str],
    ) -> None:
        """Invalid quality_requirement should return 422."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/finops/routing/optimize",
                headers=auth_headers,
                json={
                    "workload_name": "test-agent",
                    "use_case": "code_generation",
                    "quality_requirement": "ultra",  # Not allowed
                },
            )
        assert response.status_code == 422

    def test_routing_request_requires_positive_monthly_requests(
        self,
        auth_headers: dict[str, str],
    ) -> None:
        """estimated_monthly_requests <= 0 should return 422."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/finops/routing/optimize",
                headers=auth_headers,
                json={
                    "workload_name": "test-agent",
                    "use_case": "code_generation",
                    "estimated_monthly_requests": 0,  # Invalid
                },
            )
        assert response.status_code == 422
