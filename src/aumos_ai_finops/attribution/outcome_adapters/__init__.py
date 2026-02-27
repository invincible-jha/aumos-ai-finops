"""Outcome adapter package for pluggable business outcome event sources.

Outcome adapters are responsible for fetching business outcome events from
external sources and normalizing them into the OutcomeEvent dataclass that
CostOutcomeAttributor can consume.

Available adapters:
    WebhookOutcomeAdapter  — Receives outcomes via HTTP webhook POST
    ManualEntryOutcomeAdapter — In-memory/database manual entry

All adapters implement the IOutcomeAdapter protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class OutcomeEvent:
    """Normalized business outcome event consumed by CostOutcomeAttributor.

    Attributes:
        decision_id: UUID of the AI decision this outcome belongs to.
        tenant_id: Owning tenant.
        outcome_type: Categorical outcome type.
        outcome_value_usd: Monetized value in USD.
        outcome_at: When the outcome was realized.
        source: Adapter source label for audit purposes.
    """

    decision_id: str
    tenant_id: str
    outcome_type: str
    outcome_value_usd: Decimal
    outcome_at: datetime
    source: str


class IOutcomeAdapter:
    """Protocol interface for outcome event sources.

    Implementations must be async-safe and return a list of OutcomeEvent
    objects from whatever source they connect to (webhook queue, database,
    CSV, API, etc.).
    """

    async def pull_outcomes(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[OutcomeEvent]:
        """Pull pending outcome events for a tenant.

        Args:
            tenant_id: Tenant to pull outcomes for.
            limit: Maximum events to return.

        Returns:
            List of normalized OutcomeEvent objects.
        """
        raise NotImplementedError


__all__ = ["IOutcomeAdapter", "OutcomeEvent"]
