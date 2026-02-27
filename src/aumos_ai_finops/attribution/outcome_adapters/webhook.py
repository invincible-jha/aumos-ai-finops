"""WebhookOutcomeAdapter — receives business outcomes via HTTP webhook.

Outcome events arrive as POST payloads to /api/v1/finops/outcomes. This adapter
maintains an in-process pending queue populated by the API endpoint and drained
by CostOutcomeAttributor.process_pending_outcomes().

In production the queue should be backed by Redis or a database table for
durability across restarts. This implementation uses an asyncio.Queue for
correctness in single-process deployments; the repository layer should be
substituted for multi-instance deployments.

Thread safety: asyncio.Queue is safe within a single event loop thread.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from aumos_ai_finops.attribution.outcome_adapters import IOutcomeAdapter, OutcomeEvent

logger = structlog.get_logger(__name__)


class WebhookOutcomeAdapter(IOutcomeAdapter):
    """Outcome adapter that drains an in-process webhook queue.

    Webhook POST payloads are enqueued via enqueue_outcome() and drained
    by pull_outcomes(). This decouples the HTTP request path from the
    attribution processing path.

    Args:
        max_queue_size: Maximum pending events before backpressure is applied
            (0 = unlimited).
    """

    def __init__(self, max_queue_size: int = 10_000) -> None:
        """Initialise the adapter with an empty asyncio queue."""
        self._queue: asyncio.Queue[OutcomeEvent] = asyncio.Queue(maxsize=max_queue_size)

    async def enqueue_outcome(
        self,
        decision_id: str,
        tenant_id: str,
        outcome_type: str,
        outcome_value_usd: Decimal,
        outcome_at: datetime | None = None,
    ) -> None:
        """Enqueue a new outcome event from an incoming webhook POST.

        Called directly by the FastAPI endpoint handler. Does not block —
        raises asyncio.QueueFull if the queue is at capacity.

        Args:
            decision_id: UUID of the attributed AI decision.
            tenant_id: Owning tenant.
            outcome_type: Categorical outcome type.
            outcome_value_usd: Monetized USD value of the outcome.
            outcome_at: When the outcome was realized (defaults to now UTC).

        Raises:
            asyncio.QueueFull: When the queue has reached max_queue_size.
        """
        if outcome_at is None:
            outcome_at = datetime.now(timezone.utc)

        event = OutcomeEvent(
            decision_id=decision_id,
            tenant_id=tenant_id,
            outcome_type=outcome_type,
            outcome_value_usd=outcome_value_usd,
            outcome_at=outcome_at,
            source="webhook",
        )
        self._queue.put_nowait(event)

        logger.debug(
            "outcome_enqueued",
            decision_id=decision_id,
            tenant_id=tenant_id,
            outcome_type=outcome_type,
            queue_size=self._queue.qsize(),
        )

    async def pull_outcomes(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[OutcomeEvent]:
        """Drain up to limit outcome events for the given tenant.

        Events for other tenants are re-queued at the back. This is a
        best-effort per-tenant drain — in high-throughput scenarios,
        consider partitioning by tenant_id.

        Args:
            tenant_id: Tenant to pull outcomes for.
            limit: Maximum events to return.

        Returns:
            List of OutcomeEvent objects for the requested tenant.
        """
        result: list[OutcomeEvent] = []
        requeue: list[OutcomeEvent] = []
        checked = 0
        queue_snapshot_size = self._queue.qsize()

        while checked < queue_snapshot_size and len(result) < limit:
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            checked += 1
            if event.tenant_id == tenant_id:
                result.append(event)
            else:
                requeue.append(event)

        # Re-queue events that belong to other tenants
        for event in requeue:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "webhook_queue_full_on_requeue",
                    decision_id=event.decision_id,
                    tenant_id=event.tenant_id,
                )

        logger.debug(
            "webhook_outcomes_pulled",
            tenant_id=tenant_id,
            pulled=len(result),
            requeued=len(requeue),
        )
        return result

    @property
    def queue_size(self) -> int:
        """Current number of pending events in the queue."""
        return self._queue.qsize()


__all__ = ["WebhookOutcomeAdapter"]
