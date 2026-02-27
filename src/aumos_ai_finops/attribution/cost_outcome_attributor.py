"""CostOutcomeAttributor — joins AI inference costs with business outcome events.

This is the core attribution engine for P3.1: AI Cost-to-Outcome Attribution.
It collects six cost components per decision, joins them with business outcome
events (up to 90-day attribution window), computes ROI, and performs break-even
analysis per use case.

ROI formula:
    roi_pct = (outcome_value_usd - total_cost_usd) / total_cost_usd * 100

Break-even:
    days_to_break_even = total_cost_usd / daily_outcome_rate
    (None when daily_outcome_rate == 0 — decision has not yet generated value)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

from aumos_ai_finops.attribution.outcome_adapters import IOutcomeAdapter

logger = structlog.get_logger(__name__)

# Maximum look-back window for joining outcomes to a decision (90 days in seconds)
_ATTRIBUTION_WINDOW_SECONDS: int = 90 * 24 * 3600


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionCostComponents:
    """Six cost components for a single AI decision.

    All monetary fields are DECIMAL to prevent floating-point accumulation errors.

    Attributes:
        decision_id: Unique identifier for the AI decision.
        tenant_id: Tenant that owns the decision.
        ai_system_id: AI system that produced the decision.
        use_case: Business use case label.
        input_token_cost: Cost of input (prompt) tokens.
        output_token_cost: Cost of output (completion) tokens.
        compute_cost: GPU/CPU compute cost during inference.
        storage_cost: Vector store / artifact storage cost attributed to this decision.
        egress_cost: Network egress cost for response delivery.
        human_review_cost: Cost of human review if applicable (0 if no review).
        decision_at: Timestamp when the decision was made.
    """

    decision_id: str
    tenant_id: str
    ai_system_id: str
    use_case: str
    input_token_cost: Decimal
    output_token_cost: Decimal
    compute_cost: Decimal
    storage_cost: Decimal
    egress_cost: Decimal
    human_review_cost: Decimal
    decision_at: datetime

    @property
    def total_cost(self) -> Decimal:
        """Sum of all six cost components."""
        return (
            self.input_token_cost
            + self.output_token_cost
            + self.compute_cost
            + self.storage_cost
            + self.egress_cost
            + self.human_review_cost
        )


@dataclass(frozen=True)
class OutcomeAttribution:
    """Attribution result linking a decision's costs to a business outcome.

    Attributes:
        decision_id: The attributed decision.
        tenant_id: Owning tenant.
        ai_system_id: AI system that made the decision.
        use_case: Business use case.
        costs: Decomposed cost components.
        total_cost_usd: Sum of all cost components.
        outcome_type: Category of outcome (or None if no outcome attributed yet).
        outcome_value_usd: Monetized outcome value (or None if pending).
        roi_pct: Return on investment percentage (None if no outcome yet).
        attribution_complete: True when an outcome has been matched.
        decision_at: When the decision was made.
        outcome_at: When the outcome was realized (None if not yet attributed).
    """

    decision_id: str
    tenant_id: str
    ai_system_id: str
    use_case: str
    costs: DecisionCostComponents
    total_cost_usd: Decimal
    outcome_type: str | None
    outcome_value_usd: Decimal | None
    roi_pct: Decimal | None
    attribution_complete: bool
    decision_at: datetime
    outcome_at: datetime | None


@dataclass(frozen=True)
class UseCaseROISummary:
    """Aggregated ROI summary for a use case over a reporting period.

    Attributes:
        tenant_id: Owning tenant.
        use_case: Business use case label.
        period_date: Date of this summary (truncated to day).
        total_decisions: Total decisions processed.
        total_cost_usd: Sum of all decision costs.
        total_outcome_value_usd: Sum of all attributed outcome values.
        avg_roi_pct: Average ROI across decisions with outcomes.
        decisions_with_outcome: Number of decisions that have attributed outcomes.
    """

    tenant_id: str
    use_case: str
    period_date: datetime
    total_decisions: int
    total_cost_usd: Decimal
    total_outcome_value_usd: Decimal
    avg_roi_pct: Decimal
    decisions_with_outcome: int


@dataclass(frozen=True)
class BreakEvenAnalysis:
    """Break-even analysis for a use case.

    Attributes:
        use_case: Use case label.
        tenant_id: Owning tenant.
        total_cost_usd: Total investment cost to date.
        total_outcome_value_usd: Total outcome value realized to date.
        net_value_usd: Outcome value minus cost.
        roi_pct: Return on investment percentage.
        is_break_even: True when outcome_value >= total_cost.
        days_to_break_even: Projected days from first decision to break-even
            (None when daily outcome rate is zero or unknown).
    """

    use_case: str
    tenant_id: str
    total_cost_usd: Decimal
    total_outcome_value_usd: Decimal
    net_value_usd: Decimal
    roi_pct: Decimal
    is_break_even: bool
    days_to_break_even: int | None


# ---------------------------------------------------------------------------
# Repository protocol
# ---------------------------------------------------------------------------


class IAttributionRepository:
    """Protocol for cost-outcome attribution persistence.

    All methods are async and tenant-scoped.
    """

    async def save_attribution(self, attribution: OutcomeAttribution) -> None:
        """Persist or update an attribution record.

        Args:
            attribution: The attribution result to persist.
        """
        ...

    async def get_by_decision_id(
        self,
        decision_id: str,
        tenant_id: str,
    ) -> OutcomeAttribution | None:
        """Retrieve attribution for a specific decision.

        Args:
            decision_id: The decision UUID.
            tenant_id: Owning tenant (for RLS enforcement).

        Returns:
            OutcomeAttribution if found, None otherwise.
        """
        ...

    async def list_pending_attributions(
        self,
        tenant_id: str,
        decision_after: datetime,
        limit: int = 100,
    ) -> list[OutcomeAttribution]:
        """List attributions that do not yet have an outcome.

        Args:
            tenant_id: Owning tenant.
            decision_after: Only consider decisions made after this timestamp.
            limit: Maximum results to return.

        Returns:
            List of incomplete attributions within the attribution window.
        """
        ...

    async def get_use_case_summary(
        self,
        tenant_id: str,
        use_case: str,
        period_start: datetime,
        period_end: datetime,
    ) -> UseCaseROISummary | None:
        """Load aggregated ROI summary for a use case and period.

        Args:
            tenant_id: Owning tenant.
            use_case: Use case to summarize.
            period_start: Period start.
            period_end: Period end.

        Returns:
            UseCaseROISummary if data exists, None otherwise.
        """
        ...

    async def list_attributions_for_dashboard(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> list[OutcomeAttribution]:
        """Load all attributions for the ROI dashboard.

        Args:
            tenant_id: Owning tenant.
            period_start: Dashboard period start.
            period_end: Dashboard period end.

        Returns:
            All attributions in the window.
        """
        ...


# ---------------------------------------------------------------------------
# CostOutcomeAttributor
# ---------------------------------------------------------------------------


class CostOutcomeAttributor:
    """Joins AI inference costs with business outcome events to compute ROI.

    Core engine for P3.1: AI Cost-to-Outcome Attribution. Responsible for:
      - Recording 6 cost components per AI decision
      - Joining outcomes arriving within the 90-day attribution window
      - Computing ROI = (outcome_value - total_cost) / total_cost * 100
      - Performing break-even analysis per use case

    The attributor is stateless with respect to request handling. All
    persistence is delegated to the injected repository. Outcome adapters
    (webhook, manual) feed outcome events into this class.

    Args:
        attribution_repo: Repository for persisting attribution records.
        outcome_adapter: Adapter that supplies outcome events.
        attribution_window_days: Days within which an outcome can be linked
            to a decision (default 90 days).
    """

    def __init__(
        self,
        attribution_repo: IAttributionRepository,
        outcome_adapter: IOutcomeAdapter,
        attribution_window_days: int = 90,
    ) -> None:
        """Initialise the attributor with all required dependencies."""
        self._repo = attribution_repo
        self._adapter = outcome_adapter
        self._attribution_window_days = attribution_window_days

    # ------------------------------------------------------------------
    # Recording cost components
    # ------------------------------------------------------------------

    async def record_decision_costs(
        self,
        decision_id: str,
        tenant_id: str,
        ai_system_id: str,
        use_case: str,
        input_token_cost: Decimal,
        output_token_cost: Decimal,
        compute_cost: Decimal,
        storage_cost: Decimal,
        egress_cost: Decimal,
        human_review_cost: Decimal,
        decision_at: datetime | None = None,
    ) -> OutcomeAttribution:
        """Record the six cost components for an AI decision.

        Creates an incomplete attribution record (attribution_complete=False).
        The record will be completed when an outcome event arrives within
        the attribution window.

        Args:
            decision_id: Unique identifier for the AI decision (UUID string).
            tenant_id: Tenant that owns this decision.
            ai_system_id: The AI system that produced the decision.
            use_case: Business use case label for grouping.
            input_token_cost: Cost of prompt/input tokens in USD.
            output_token_cost: Cost of completion/output tokens in USD.
            compute_cost: GPU/CPU compute cost during inference in USD.
            storage_cost: Storage cost attributed to this decision in USD.
            egress_cost: Network egress cost for response delivery in USD.
            human_review_cost: Cost of human review if any (0 if none) in USD.
            decision_at: When the decision was made (defaults to now UTC).

        Returns:
            Incomplete OutcomeAttribution record (no outcome yet).
        """
        if decision_at is None:
            decision_at = datetime.now(timezone.utc)

        costs = DecisionCostComponents(
            decision_id=decision_id,
            tenant_id=tenant_id,
            ai_system_id=ai_system_id,
            use_case=use_case,
            input_token_cost=input_token_cost,
            output_token_cost=output_token_cost,
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            egress_cost=egress_cost,
            human_review_cost=human_review_cost,
            decision_at=decision_at,
        )

        attribution = OutcomeAttribution(
            decision_id=decision_id,
            tenant_id=tenant_id,
            ai_system_id=ai_system_id,
            use_case=use_case,
            costs=costs,
            total_cost_usd=costs.total_cost,
            outcome_type=None,
            outcome_value_usd=None,
            roi_pct=None,
            attribution_complete=False,
            decision_at=decision_at,
            outcome_at=None,
        )

        await self._repo.save_attribution(attribution)

        logger.info(
            "decision_costs_recorded",
            decision_id=decision_id,
            tenant_id=tenant_id,
            ai_system_id=ai_system_id,
            use_case=use_case,
            total_cost_usd=str(costs.total_cost),
        )

        return attribution

    # ------------------------------------------------------------------
    # Attributing outcomes
    # ------------------------------------------------------------------

    async def attribute_outcome(
        self,
        decision_id: str,
        tenant_id: str,
        outcome_type: str,
        outcome_value_usd: Decimal,
        outcome_at: datetime | None = None,
    ) -> OutcomeAttribution:
        """Link a business outcome to a previously recorded AI decision.

        Computes ROI and marks the attribution as complete. If the decision
        has not been recorded or the attribution window has expired, raises
        ValueError.

        Args:
            decision_id: The AI decision to attribute the outcome to.
            tenant_id: Owning tenant (enforces RLS).
            outcome_type: Category of outcome (e.g., revenue_generated).
            outcome_value_usd: Monetized value of the outcome in USD.
            outcome_at: When the outcome was realized (defaults to now UTC).

        Returns:
            Completed OutcomeAttribution with ROI calculated.

        Raises:
            ValueError: If the decision is not found or the attribution
                window has expired.
        """
        if outcome_at is None:
            outcome_at = datetime.now(timezone.utc)

        existing = await self._repo.get_by_decision_id(decision_id, tenant_id)
        if existing is None:
            raise ValueError(
                f"No cost record found for decision '{decision_id}' in tenant '{tenant_id}'. "
                "Record costs before attributing an outcome."
            )

        # Enforce attribution window
        window_cutoff = existing.decision_at.replace(tzinfo=timezone.utc) if existing.decision_at.tzinfo is None else existing.decision_at
        elapsed_seconds = (outcome_at - window_cutoff).total_seconds()
        window_seconds = self._attribution_window_days * 86_400
        if elapsed_seconds > window_seconds:
            raise ValueError(
                f"Attribution window of {self._attribution_window_days} days has expired "
                f"for decision '{decision_id}'. Decision made at {existing.decision_at.isoformat()}, "
                f"outcome at {outcome_at.isoformat()}."
            )

        roi_pct = _compute_roi(existing.total_cost_usd, outcome_value_usd)

        completed = OutcomeAttribution(
            decision_id=existing.decision_id,
            tenant_id=existing.tenant_id,
            ai_system_id=existing.ai_system_id,
            use_case=existing.use_case,
            costs=existing.costs,
            total_cost_usd=existing.total_cost_usd,
            outcome_type=outcome_type,
            outcome_value_usd=outcome_value_usd,
            roi_pct=roi_pct,
            attribution_complete=True,
            decision_at=existing.decision_at,
            outcome_at=outcome_at,
        )

        await self._repo.save_attribution(completed)

        logger.info(
            "outcome_attributed",
            decision_id=decision_id,
            tenant_id=tenant_id,
            outcome_type=outcome_type,
            outcome_value_usd=str(outcome_value_usd),
            total_cost_usd=str(existing.total_cost_usd),
            roi_pct=str(roi_pct),
        )

        return completed

    # ------------------------------------------------------------------
    # Querying attribution results
    # ------------------------------------------------------------------

    async def get_decision_roi(
        self,
        decision_id: str,
        tenant_id: str,
    ) -> OutcomeAttribution:
        """Retrieve the cost breakdown and ROI for a single AI decision.

        Args:
            decision_id: The decision UUID.
            tenant_id: Owning tenant.

        Returns:
            OutcomeAttribution with cost components and ROI (if attributed).

        Raises:
            ValueError: If no attribution record exists for this decision.
        """
        result = await self._repo.get_by_decision_id(decision_id, tenant_id)
        if result is None:
            raise ValueError(
                f"No attribution record found for decision '{decision_id}' "
                f"in tenant '{tenant_id}'."
            )
        return result

    async def get_use_case_roi_summary(
        self,
        tenant_id: str,
        use_case: str,
        period_start: datetime,
        period_end: datetime,
    ) -> UseCaseROISummary | None:
        """Get aggregated ROI summary for a use case over a period.

        Args:
            tenant_id: Owning tenant.
            use_case: Business use case to summarize.
            period_start: Period start (UTC).
            period_end: Period end (UTC).

        Returns:
            UseCaseROISummary or None if no data exists.
        """
        return await self._repo.get_use_case_summary(
            tenant_id=tenant_id,
            use_case=use_case,
            period_start=period_start,
            period_end=period_end,
        )

    async def get_roi_dashboard(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> dict[str, Any]:
        """Build the full ROI dashboard payload for a tenant.

        Aggregates all attributions in the period into per-use-case summaries,
        computes break-even analysis, and calculates overall portfolio ROI.

        Args:
            tenant_id: Owning tenant.
            period_start: Dashboard period start (UTC).
            period_end: Dashboard period end (UTC).

        Returns:
            Dashboard dict with keys: attributions, use_case_summaries,
            break_even_analyses, portfolio_roi_pct, total_cost_usd,
            total_outcome_value_usd, decisions_with_outcome.
        """
        attributions = await self._repo.list_attributions_for_dashboard(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
        )

        # Aggregate by use case
        by_use_case: dict[str, list[OutcomeAttribution]] = {}
        for attr in attributions:
            by_use_case.setdefault(attr.use_case, []).append(attr)

        use_case_summaries: list[UseCaseROISummary] = []
        break_even_analyses: list[BreakEvenAnalysis] = []

        for use_case, attrs in by_use_case.items():
            total_decisions = len(attrs)
            total_cost = sum(a.total_cost_usd for a in attrs)
            completed = [a for a in attrs if a.attribution_complete and a.outcome_value_usd is not None]
            total_outcome = sum(a.outcome_value_usd for a in completed if a.outcome_value_usd)  # type: ignore[misc]
            decisions_with_outcome = len(completed)
            avg_roi = _compute_roi(total_cost, total_outcome) if total_cost > 0 and total_outcome is not None else Decimal("0")

            summary = UseCaseROISummary(
                tenant_id=tenant_id,
                use_case=use_case,
                period_date=period_start,
                total_decisions=total_decisions,
                total_cost_usd=total_cost,
                total_outcome_value_usd=total_outcome,
                avg_roi_pct=avg_roi,
                decisions_with_outcome=decisions_with_outcome,
            )
            use_case_summaries.append(summary)

            bea = _compute_break_even(use_case, tenant_id, total_cost, total_outcome, attrs)
            break_even_analyses.append(bea)

        # Portfolio totals
        total_cost_portfolio = sum(a.total_cost_usd for a in attributions)
        total_outcome_portfolio = sum(
            a.outcome_value_usd for a in attributions
            if a.attribution_complete and a.outcome_value_usd is not None
        )
        portfolio_roi = (
            _compute_roi(total_cost_portfolio, total_outcome_portfolio)
            if total_cost_portfolio > 0
            else Decimal("0")
        )
        decisions_with_outcome_total = sum(
            1 for a in attributions if a.attribution_complete and a.outcome_value_usd is not None
        )

        logger.info(
            "roi_dashboard_computed",
            tenant_id=tenant_id,
            total_decisions=len(attributions),
            decisions_with_outcome=decisions_with_outcome_total,
            portfolio_roi_pct=str(portfolio_roi),
        )

        return {
            "tenant_id": tenant_id,
            "period_start": period_start,
            "period_end": period_end,
            "attributions": attributions,
            "use_case_summaries": use_case_summaries,
            "break_even_analyses": break_even_analyses,
            "portfolio_roi_pct": portfolio_roi,
            "total_cost_usd": total_cost_portfolio,
            "total_outcome_value_usd": total_outcome_portfolio,
            "decisions_with_outcome": decisions_with_outcome_total,
            "total_decisions": len(attributions),
        }

    # ------------------------------------------------------------------
    # Outcome adapter integration
    # ------------------------------------------------------------------

    async def process_pending_outcomes(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> int:
        """Pull pending outcome events from the adapter and attribute them.

        Polls the configured outcome adapter for new outcome events and
        attempts to attribute each to its corresponding decision record.
        Events with expired attribution windows or missing decision records
        are logged and skipped.

        Args:
            tenant_id: Tenant to process outcomes for.
            limit: Maximum outcomes to process in one pass.

        Returns:
            Number of outcomes successfully attributed.
        """
        from aumos_ai_finops.attribution.outcome_adapters import OutcomeEvent

        events: list[OutcomeEvent] = await self._adapter.pull_outcomes(
            tenant_id=tenant_id,
            limit=limit,
        )
        attributed_count = 0

        for event in events:
            try:
                await self.attribute_outcome(
                    decision_id=event.decision_id,
                    tenant_id=tenant_id,
                    outcome_type=event.outcome_type,
                    outcome_value_usd=event.outcome_value_usd,
                    outcome_at=event.outcome_at,
                )
                attributed_count += 1
            except ValueError as exc:
                logger.warning(
                    "outcome_attribution_skipped",
                    decision_id=event.decision_id,
                    tenant_id=tenant_id,
                    reason=str(exc),
                )

        logger.info(
            "pending_outcomes_processed",
            tenant_id=tenant_id,
            total_events=len(events),
            attributed=attributed_count,
        )
        return attributed_count


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_roi(total_cost: Decimal, outcome_value: Decimal) -> Decimal:
    """Compute ROI percentage.

    Formula: (outcome_value - total_cost) / total_cost * 100

    Returns 0 when total_cost is zero to avoid division by zero.

    Args:
        total_cost: Total cost of the AI decision(s).
        outcome_value: Total realized outcome value in USD.

    Returns:
        ROI as a Decimal percentage (may be negative for negative ROI).
    """
    if total_cost == Decimal("0"):
        return Decimal("0")
    return ((outcome_value - total_cost) / total_cost * Decimal("100")).quantize(Decimal("0.01"))


def _compute_break_even(
    use_case: str,
    tenant_id: str,
    total_cost: Decimal,
    total_outcome: Decimal,
    attributions: list[OutcomeAttribution],
) -> BreakEvenAnalysis:
    """Compute break-even analysis for a use case.

    Estimates days to break even based on average daily outcome rate
    derived from decisions that have attributed outcomes.

    Args:
        use_case: Use case label.
        tenant_id: Owning tenant.
        total_cost: Total cost across all decisions in the period.
        total_outcome: Total outcome value across completed attributions.
        attributions: All attributions for this use case in the period.

    Returns:
        BreakEvenAnalysis with projected days to break even.
    """
    net_value = total_outcome - total_cost
    roi_pct = _compute_roi(total_cost, total_outcome)
    is_break_even = total_outcome >= total_cost

    # Estimate daily outcome rate from completed attributions
    days_to_break_even: int | None = None
    completed = [
        a for a in attributions
        if a.attribution_complete and a.outcome_value_usd is not None and a.outcome_at is not None
    ]
    if completed and total_cost > 0 and not is_break_even:
        # Compute total elapsed days across completed attributions
        total_elapsed_days = sum(
            max(1, (a.outcome_at - a.decision_at).days)  # type: ignore[operator]
            for a in completed
        )
        if total_elapsed_days > 0 and total_outcome > 0:
            # Average daily outcome rate
            daily_rate = total_outcome / Decimal(str(total_elapsed_days))
            if daily_rate > 0:
                days_to_break_even = int((total_cost / daily_rate).to_integral_value())

    return BreakEvenAnalysis(
        use_case=use_case,
        tenant_id=tenant_id,
        total_cost_usd=total_cost,
        total_outcome_value_usd=total_outcome,
        net_value_usd=net_value,
        roi_pct=roi_pct,
        is_break_even=is_break_even,
        days_to_break_even=days_to_break_even,
    )


__all__ = [
    "CostOutcomeAttributor",
    "DecisionCostComponents",
    "OutcomeAttribution",
    "UseCaseROISummary",
    "BreakEvenAnalysis",
    "IAttributionRepository",
]
