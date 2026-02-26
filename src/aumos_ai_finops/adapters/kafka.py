"""Kafka event publishers for the AumOS AI FinOps service.

Publishes domain events after state changes in the FinOps system:
  - finops.cost_recorded    — a new cost record has been persisted
  - finops.budget_exceeded  — a budget threshold has been breached
  - finops.roi_calculated   — an ROI calculation has completed
"""

import uuid
from datetime import datetime

from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class FinOpsEventPublisher:
    """Publishes FinOps domain events to Kafka.

    Wraps the aumos-common EventPublisher to provide strongly-typed
    publish methods specific to the FinOps domain.
    """

    def __init__(self, publisher: EventPublisher) -> None:
        """Initialize with the underlying aumos-common EventPublisher."""
        self._publisher = publisher

    async def publish_cost_recorded(
        self,
        tenant_id: str,
        cost_record_id: str,
        resource_type: str,
        cost_usd: float,
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """Publish a finops.cost_recorded event.

        Args:
            tenant_id: The tenant this cost belongs to.
            cost_record_id: UUID of the persisted CostRecord.
            resource_type: Resource type (gpu | cpu | storage | etc.).
            cost_usd: Total cost in USD.
            period_start: Billing window start.
            period_end: Billing window end.
        """
        correlation_id = str(uuid.uuid4())
        payload = {
            "event_type": "cost_recorded",
            "tenant_id": tenant_id,
            "cost_record_id": cost_record_id,
            "resource_type": resource_type,
            "cost_usd": cost_usd,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "correlation_id": correlation_id,
        }
        await self._publisher.publish(
            topic=Topics.FINOPS_COST_RECORDED,
            payload=payload,
            key=tenant_id,
        )
        logger.debug(
            "kafka_event_published",
            topic=Topics.FINOPS_COST_RECORDED,
            tenant_id=tenant_id,
            cost_record_id=cost_record_id,
            correlation_id=correlation_id,
        )

    async def publish_budget_exceeded(
        self,
        tenant_id: str,
        budget_id: str,
        budget_name: str,
        severity: str,
        actual_spend_usd: float,
        limit_usd: float,
        utilization_percent: float,
    ) -> None:
        """Publish a finops.budget_exceeded event.

        Args:
            tenant_id: The tenant whose budget was exceeded.
            budget_id: UUID of the Budget record.
            budget_name: Human-readable budget name.
            severity: warning | critical
            actual_spend_usd: Actual spend at trigger time.
            limit_usd: Configured budget limit.
            utilization_percent: Percentage of budget consumed.
        """
        correlation_id = str(uuid.uuid4())
        payload = {
            "event_type": "budget_exceeded",
            "tenant_id": tenant_id,
            "budget_id": budget_id,
            "budget_name": budget_name,
            "severity": severity,
            "actual_spend_usd": actual_spend_usd,
            "limit_usd": limit_usd,
            "utilization_percent": utilization_percent,
            "correlation_id": correlation_id,
        }
        await self._publisher.publish(
            topic=Topics.FINOPS_BUDGET_EXCEEDED,
            payload=payload,
            key=tenant_id,
        )
        logger.warning(
            "kafka_event_published",
            topic=Topics.FINOPS_BUDGET_EXCEEDED,
            tenant_id=tenant_id,
            budget_id=budget_id,
            severity=severity,
            utilization_percent=utilization_percent,
            correlation_id=correlation_id,
        )

    async def publish_roi_calculated(
        self,
        tenant_id: str,
        calculation_id: str,
        initiative_name: str,
        roi_percent: float,
        total_benefit_usd: float,
        total_ai_cost_usd: float,
    ) -> None:
        """Publish a finops.roi_calculated event.

        Args:
            tenant_id: The tenant for which ROI was calculated.
            calculation_id: UUID of the ROICalculation record.
            initiative_name: Name of the AI initiative.
            roi_percent: Calculated ROI percentage.
            total_benefit_usd: Total monetary benefit.
            total_ai_cost_usd: Total AI investment cost.
        """
        correlation_id = str(uuid.uuid4())
        payload = {
            "event_type": "roi_calculated",
            "tenant_id": tenant_id,
            "calculation_id": calculation_id,
            "initiative_name": initiative_name,
            "roi_percent": roi_percent,
            "total_benefit_usd": total_benefit_usd,
            "total_ai_cost_usd": total_ai_cost_usd,
            "correlation_id": correlation_id,
        }
        await self._publisher.publish(
            topic=Topics.FINOPS_ROI_CALCULATED,
            payload=payload,
            key=tenant_id,
        )
        logger.info(
            "kafka_event_published",
            topic=Topics.FINOPS_ROI_CALCULATED,
            tenant_id=tenant_id,
            calculation_id=calculation_id,
            roi_percent=roi_percent,
            correlation_id=correlation_id,
        )
