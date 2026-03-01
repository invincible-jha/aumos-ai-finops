"""Cost anomaly detection adapter for aumos-ai-finops.

Detects unusual spending patterns by comparing current period costs against
rolling baseline statistics. Triggers budget alerts when anomalous cost spikes
are detected, enabling proactive cost governance before budgets are exhausted.

Gap Coverage: GAP-265 (Cost Anomaly Detection)
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Minimum number of data points before anomaly detection is reliable
_MIN_BASELINE_SAMPLES = 7


@dataclass
class CostAnomalySignal:
    """An anomalous cost pattern detected for a tenant.

    Attributes:
        tenant_id: The tenant with the anomalous cost pattern.
        model_id: Model ID with the anomaly (None = aggregate anomaly).
        current_cost_usd: Current period cost.
        baseline_mean_usd: Rolling mean from baseline.
        baseline_stddev_usd: Rolling standard deviation from baseline.
        z_score: Deviation in standard deviations above the mean.
        percent_above_baseline: How much above baseline as percentage.
        severity: warning | critical.
        detected_at: When the anomaly was detected.
    """

    tenant_id: uuid.UUID
    model_id: str | None
    current_cost_usd: float
    baseline_mean_usd: float
    baseline_stddev_usd: float
    z_score: float
    percent_above_baseline: float
    severity: str
    detected_at: datetime


class CostAnomalyDetector:
    """Statistical anomaly detector for AI spending patterns.

    Uses rolling baseline statistics (mean, stddev) to identify cost spikes
    that exceed configured z-score thresholds. Integrates with the budget
    alert system to trigger notifications when anomalies are detected.

    Args:
        baseline_repository: Repository for cost baseline statistics.
        event_publisher: FinOps event publisher for anomaly alerts.
        warn_z_score: Z-score threshold for warning signals (default: 2.0).
        critical_z_score: Z-score threshold for critical signals (default: 3.5).
        min_cost_threshold_usd: Ignore anomalies below this USD amount (noise floor).
    """

    def __init__(
        self,
        baseline_repository: Any,
        event_publisher: Any,
        warn_z_score: float = 2.0,
        critical_z_score: float = 3.5,
        min_cost_threshold_usd: float = 1.0,
    ) -> None:
        """Initialize the cost anomaly detector.

        Args:
            baseline_repository: Repository for cost baseline data.
            event_publisher: FinOps event publisher.
            warn_z_score: Warning threshold as z-score.
            critical_z_score: Critical threshold as z-score.
            min_cost_threshold_usd: Minimum cost delta to trigger an anomaly.
        """
        self._baseline_repo = baseline_repository
        self._event_publisher = event_publisher
        self._warn_z_score = warn_z_score
        self._critical_z_score = critical_z_score
        self._min_cost_threshold = min_cost_threshold_usd

    async def check_for_anomalies(
        self,
        tenant_id: uuid.UUID,
        current_costs: dict[str, float],
    ) -> list[CostAnomalySignal]:
        """Check current cost data for anomalous patterns.

        Compares current period costs against stored baselines.
        Returns anomaly signals for all metrics exceeding thresholds.

        Args:
            tenant_id: The tenant to evaluate.
            current_costs: Dict mapping metric_key -> current_cost_usd.
                Keys: "total" for aggregate, "model:{model_id}" for per-model.

        Returns:
            List of CostAnomalySignal for all detected anomalies.
        """
        signals: list[CostAnomalySignal] = []
        now = datetime.now(tz=timezone.utc)

        for metric_key, current_cost in current_costs.items():
            model_id = metric_key.split(":", 1)[1] if ":" in metric_key else None

            try:
                baseline = await self._baseline_repo.get_cost_baseline(
                    tenant_id=tenant_id,
                    metric_key=metric_key,
                )
            except Exception as exc:
                logger.warning(
                    "Could not retrieve cost baseline",
                    tenant_id=str(tenant_id),
                    metric_key=metric_key,
                    error=str(exc),
                )
                continue

            if baseline is None or baseline.get("sample_count", 0) < _MIN_BASELINE_SAMPLES:
                logger.debug(
                    "Insufficient baseline samples for anomaly detection",
                    tenant_id=str(tenant_id),
                    metric_key=metric_key,
                    sample_count=baseline.get("sample_count", 0) if baseline else 0,
                )
                continue

            mean = float(baseline["mean_usd"])
            stddev = float(baseline["stddev_usd"])

            # Skip if cost is below noise floor
            cost_delta = current_cost - mean
            if abs(cost_delta) < self._min_cost_threshold:
                continue

            if stddev == 0.0:
                z_score = float("inf") if cost_delta > 0 else 0.0
            else:
                z_score = cost_delta / stddev

            # Only flag upward anomalies (cost increases, not drops)
            if z_score < self._warn_z_score:
                continue

            percent_above = (cost_delta / mean * 100) if mean > 0 else 0.0
            severity = "critical" if z_score >= self._critical_z_score else "warning"

            signal = CostAnomalySignal(
                tenant_id=tenant_id,
                model_id=model_id,
                current_cost_usd=current_cost,
                baseline_mean_usd=mean,
                baseline_stddev_usd=stddev,
                z_score=round(z_score, 3),
                percent_above_baseline=round(percent_above, 1),
                severity=severity,
                detected_at=now,
            )
            signals.append(signal)

            logger.warning(
                "Cost anomaly detected",
                tenant_id=str(tenant_id),
                metric_key=metric_key,
                current_cost_usd=round(current_cost, 4),
                baseline_mean_usd=round(mean, 4),
                z_score=round(z_score, 3),
                percent_above_baseline=round(percent_above, 1),
                severity=severity,
            )

        return signals

    async def update_baseline(
        self,
        tenant_id: uuid.UUID,
        metric_key: str,
        new_cost: float,
    ) -> None:
        """Update the rolling baseline with a new cost observation.

        Uses Welford's online algorithm for numerically stable updates.

        Args:
            tenant_id: The tenant whose baseline to update.
            metric_key: Metric identifier (total or model:{model_id}).
            new_cost: New observed cost value in USD.
        """
        try:
            baseline = await self._baseline_repo.get_cost_baseline(
                tenant_id=tenant_id,
                metric_key=metric_key,
            )

            if baseline is None:
                await self._baseline_repo.create_cost_baseline(
                    tenant_id=tenant_id,
                    metric_key=metric_key,
                    mean_usd=new_cost,
                    stddev_usd=0.0,
                    sample_count=1,
                )
                return

            n = int(baseline.get("sample_count", 0)) + 1
            old_mean = float(baseline["mean_usd"])
            old_m2 = (float(baseline["stddev_usd"]) ** 2) * max(n - 2, 1)

            delta = new_cost - old_mean
            new_mean = old_mean + delta / n
            delta2 = new_cost - new_mean
            new_m2 = old_m2 + delta * delta2
            new_stddev = (new_m2 / max(n - 1, 1)) ** 0.5

            await self._baseline_repo.update_cost_baseline(
                tenant_id=tenant_id,
                metric_key=metric_key,
                mean_usd=new_mean,
                stddev_usd=new_stddev,
                sample_count=n,
            )

        except Exception as exc:
            logger.error(
                "Failed to update cost baseline",
                tenant_id=str(tenant_id),
                metric_key=metric_key,
                error=str(exc),
            )


__all__ = [
    "CostAnomalySignal",
    "CostAnomalyDetector",
]
