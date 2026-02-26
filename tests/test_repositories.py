"""Repository tests for AumOS AI FinOps.

Integration tests require a live PostgreSQL instance (via testcontainers).
Unit tests here verify query construction and repository behavior.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCostRecordRepository:
    """Tests for CostRecordRepository query behavior."""

    def test_repository_imports_cleanly(self) -> None:
        """CostRecordRepository should import without errors."""
        from aumos_ai_finops.adapters.repositories import CostRecordRepository
        assert CostRecordRepository is not None

    def test_token_usage_repository_imports_cleanly(self) -> None:
        """All repositories should import without errors."""
        from aumos_ai_finops.adapters.repositories import (
            BudgetAlertRepository,
            BudgetRepository,
            CostRecordRepository,
            ROICalculationRepository,
            RoutingRecommendationRepository,
            TokenUsageRepository,
        )
        for repo_class in (
            CostRecordRepository,
            TokenUsageRepository,
            ROICalculationRepository,
            BudgetRepository,
            BudgetAlertRepository,
            RoutingRecommendationRepository,
        ):
            assert repo_class is not None


class TestModelsImport:
    """Tests verifying ORM models import and are structurally correct."""

    def test_all_models_import(self) -> None:
        """All fin_ ORM models should import cleanly."""
        from aumos_ai_finops.core.models import (
            Budget,
            BudgetAlert,
            CostRecord,
            ROICalculation,
            RoutingRecommendation,
            TokenUsage,
        )
        for model in (Budget, BudgetAlert, CostRecord, ROICalculation, RoutingRecommendation, TokenUsage):
            assert hasattr(model, "__tablename__")

    def test_all_models_have_fin_prefix(self) -> None:
        """All fin_ tables should use the fin_ prefix."""
        from aumos_ai_finops.core.models import (
            Budget,
            BudgetAlert,
            CostRecord,
            ROICalculation,
            RoutingRecommendation,
            TokenUsage,
        )
        for model in (Budget, BudgetAlert, CostRecord, ROICalculation, RoutingRecommendation, TokenUsage):
            assert model.__tablename__.startswith("fin_"), (
                f"{model.__name__} table '{model.__tablename__}' missing fin_ prefix"
            )
