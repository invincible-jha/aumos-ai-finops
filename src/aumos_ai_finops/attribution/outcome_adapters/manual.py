"""ManualEntryOutcomeAdapter — allows operators to manually submit outcome events.

Used when business outcomes are known but not automatically captured via
webhook (e.g., from a spreadsheet import, a CRM export, or an operator
entering a deal close value). Outcomes are persisted to the database so
they survive service restarts.

This adapter is intended for use with the POST /api/v1/finops/outcomes endpoint
when the payload source is "manual" rather than an automated system.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import structlog

from aumos_ai_finops.attribution.outcome_adapters import IOutcomeAdapter, OutcomeEvent

logger = structlog.get_logger(__name__)


class IManualOutcomeStore:
    """Protocol for persisting manual outcome entries.

    In production this wraps a SQLAlchemy async repository backed by the
    fin_manual_outcomes table (or the fin_cost_outcome_attributions table).
    """

    async def save(self, event: OutcomeEvent) -> None:
        """Persist a manual outcome event.

        Args:
            event: The outcome event to persist.
        """
        raise NotImplementedError

    async def list_pending(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[OutcomeEvent]:
        """Return unprocessed manual outcome events for a tenant.

        Args:
            tenant_id: Owning tenant.
            limit: Maximum events to return.

        Returns:
            List of pending OutcomeEvent records.
        """
        raise NotImplementedError

    async def mark_processed(self, decision_id: str, tenant_id: str) -> None:
        """Mark a manual outcome as processed so it is not returned again.

        Args:
            decision_id: The decision whose outcome was processed.
            tenant_id: Owning tenant.
        """
        raise NotImplementedError


class _InMemoryManualOutcomeStore(IManualOutcomeStore):
    """In-memory implementation for testing and single-process deployments."""

    def __init__(self) -> None:
        self._events: list[OutcomeEvent] = []
        self._processed: set[tuple[str, str]] = set()

    async def save(self, event: OutcomeEvent) -> None:
        """Store an event in memory."""
        self._events.append(event)

    async def list_pending(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[OutcomeEvent]:
        """Return stored events for the tenant that have not been processed."""
        result = [
            e
            for e in self._events
            if e.tenant_id == tenant_id and (e.decision_id, tenant_id) not in self._processed
        ]
        return result[:limit]

    async def mark_processed(self, decision_id: str, tenant_id: str) -> None:
        """Mark an event as processed."""
        self._processed.add((decision_id, tenant_id))


class ManualEntryOutcomeAdapter(IOutcomeAdapter):
    """Outcome adapter for operator-entered business outcomes.

    Outcomes are submitted via the API and stored by the injected store.
    pull_outcomes() returns all pending entries for the tenant and marks
    them processed so they are not returned on the next poll.

    Args:
        store: Persistence layer for manual outcome entries.
            Defaults to an in-memory implementation suitable for testing.
        adapter_config: Optional configuration dict (reserved for future use).
    """

    def __init__(
        self,
        store: IManualOutcomeStore | None = None,
        adapter_config: dict | None = None,
    ) -> None:
        """Initialise the adapter with a persistence store."""
        self._store: IManualOutcomeStore = store or _InMemoryManualOutcomeStore()
        self._adapter_config = adapter_config or {}

    async def submit_outcome(
        self,
        decision_id: str,
        tenant_id: str,
        outcome_type: str,
        outcome_value_usd: Decimal,
        outcome_at: datetime | None = None,
    ) -> OutcomeEvent:
        """Record a manually entered outcome event.

        Called by the POST /api/v1/finops/outcomes endpoint when source="manual".

        Args:
            decision_id: UUID of the AI decision to attribute.
            tenant_id: Owning tenant.
            outcome_type: Category of the business outcome.
            outcome_value_usd: USD value of the outcome.
            outcome_at: When the outcome occurred (defaults to now UTC).

        Returns:
            Persisted OutcomeEvent.
        """
        if outcome_at is None:
            outcome_at = datetime.now(timezone.utc)

        event = OutcomeEvent(
            decision_id=decision_id,
            tenant_id=tenant_id,
            outcome_type=outcome_type,
            outcome_value_usd=outcome_value_usd,
            outcome_at=outcome_at,
            source="manual",
        )
        await self._store.save(event)

        logger.info(
            "manual_outcome_submitted",
            decision_id=decision_id,
            tenant_id=tenant_id,
            outcome_type=outcome_type,
            outcome_value_usd=str(outcome_value_usd),
        )
        return event

    async def pull_outcomes(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[OutcomeEvent]:
        """Return and mark-processed up to limit pending manual outcomes.

        Args:
            tenant_id: Tenant to pull outcomes for.
            limit: Maximum events to return.

        Returns:
            List of pending OutcomeEvent objects for this tenant.
        """
        events = await self._store.list_pending(tenant_id=tenant_id, limit=limit)

        for event in events:
            await self._store.mark_processed(event.decision_id, tenant_id)

        logger.debug(
            "manual_outcomes_pulled",
            tenant_id=tenant_id,
            count=len(events),
        )
        return events


__all__ = [
    "ManualEntryOutcomeAdapter",
    "IManualOutcomeStore",
    "_InMemoryManualOutcomeStore",
]
