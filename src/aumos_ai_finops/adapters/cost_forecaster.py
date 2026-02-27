"""CostForecaster adapter for AI spend trend analysis and budget projection.

Analyses historical cost records to detect trends, seasonal patterns, and
budget exhaustion timelines. Generates forward-looking cost forecasts for
executive planning and budget cycle management.
"""

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Minimum data points required for trend analysis
MIN_DATA_POINTS_FOR_TREND: int = 7
MIN_DATA_POINTS_FOR_SEASONAL: int = 14

# Forecast horizon presets (days)
HORIZON_30D = 30
HORIZON_90D = 90
HORIZON_365D = 365

# Anomaly detection: Z-score threshold for flagging cost spikes
ANOMALY_ZSCORE_THRESHOLD: float = 2.5


class CostForecaster:
    """AI FinOps cost trend analysis and forward projection engine.

    Uses linear and exponential regression on historical cost time-series
    to project future spend, detect seasonal patterns, predict budget
    exhaustion dates, and flag cost anomalies. All projections are
    per-tenant and respect AumOS RLS isolation through the repository layer.

    The forecaster is a pure computation adapter — it receives historical
    data as structured dicts and produces forecast objects without touching
    the database directly (the service layer handles persistence).
    """

    def __init__(
        self,
        default_horizon_days: int = HORIZON_30D,
        anomaly_zscore_threshold: float = ANOMALY_ZSCORE_THRESHOLD,
        seasonal_smoothing_alpha: float = 0.3,
    ) -> None:
        """Initialise the CostForecaster.

        Args:
            default_horizon_days: Default forecast horizon in days.
            anomaly_zscore_threshold: Z-score above which a data point is
                flagged as anomalous.
            seasonal_smoothing_alpha: Exponential smoothing factor for
                seasonal decomposition (0 < alpha <= 1; lower = smoother).
        """
        self._default_horizon = default_horizon_days
        self._anomaly_threshold = anomaly_zscore_threshold
        self._alpha = seasonal_smoothing_alpha

    async def analyze_cost_trends(
        self,
        tenant_id: str,
        daily_costs: list[dict[str, Any]],
        resource_type: str | None = None,
    ) -> dict[str, Any]:
        """Analyse historical daily cost data to characterise the spend trend.

        Args:
            tenant_id: Tenant identifier.
            daily_costs: List of daily cost dicts with "date" (ISO str) and
                "cost_usd" (float) fields, ordered chronologically.
            resource_type: Optional resource type label for the analysis.

        Returns:
            Dict with trend_type, slope_usd_per_day, r_squared, mean_daily_cost,
            volatility_percent, trend_direction, and data_quality fields.
        """
        logger.info(
            "Analyzing cost trends",
            tenant_id=tenant_id,
            data_points=len(daily_costs),
            resource_type=resource_type,
        )

        if len(daily_costs) < MIN_DATA_POINTS_FOR_TREND:
            return {
                "tenant_id": tenant_id,
                "resource_type": resource_type,
                "trend_type": "insufficient_data",
                "data_points": len(daily_costs),
                "min_required": MIN_DATA_POINTS_FOR_TREND,
                "message": f"At least {MIN_DATA_POINTS_FOR_TREND} days of data required for trend analysis.",
            }

        costs = [float(d.get("cost_usd", 0.0)) for d in daily_costs]
        x_values = list(range(len(costs)))

        # Linear regression: y = slope * x + intercept
        slope, intercept, r_squared = self._linear_regression(x_values, costs)

        mean_cost = sum(costs) / len(costs)
        variance = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        volatility_percent = round((std_dev / mean_cost * 100.0) if mean_cost > 0 else 0.0, 2)

        # Classify trend type
        if abs(slope) < 0.01 * mean_cost:
            trend_type = "stable"
        elif slope > 0:
            # Check for exponential growth
            log_costs = [math.log(max(c, 0.001)) for c in costs]
            _, _, log_r2 = self._linear_regression(x_values, log_costs)
            trend_type = "exponential" if log_r2 > r_squared + 0.05 else "linear_growth"
        else:
            trend_type = "linear_decline"

        trend_direction = (
            "increasing" if slope > 0.01 * mean_cost
            else "decreasing" if slope < -0.01 * mean_cost
            else "stable"
        )

        result: dict[str, Any] = {
            "tenant_id": tenant_id,
            "resource_type": resource_type,
            "data_points": len(costs),
            "trend_type": trend_type,
            "slope_usd_per_day": round(slope, 4),
            "intercept_usd": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "mean_daily_cost_usd": round(mean_cost, 4),
            "std_dev_usd": round(std_dev, 4),
            "volatility_percent": volatility_percent,
            "trend_direction": trend_direction,
            "data_quality": "good" if r_squared >= 0.70 else "moderate" if r_squared >= 0.40 else "low",
            "analyzed_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "Cost trend analysis completed",
            tenant_id=tenant_id,
            trend_type=trend_type,
            slope_usd_per_day=result["slope_usd_per_day"],
            r_squared=result["r_squared"],
        )

        return result

    async def project_costs(
        self,
        tenant_id: str,
        trend_analysis: dict[str, Any],
        daily_costs: list[dict[str, Any]],
        horizon_days: int | None = None,
        projection_model: str = "linear",
    ) -> dict[str, Any]:
        """Generate a forward cost projection from trend analysis.

        Args:
            tenant_id: Tenant identifier.
            trend_analysis: Output dict from analyze_cost_trends.
            daily_costs: Historical daily cost data (used as baseline).
            horizon_days: Number of days to project forward.
            projection_model: "linear" | "exponential" | "seasonal"
                (defaults to linear unless seasonal pattern detected).

        Returns:
            Dict with projected_daily_costs (list), projected_period_total_usd,
            confidence_band_low, confidence_band_high, and model_used fields.
        """
        effective_horizon = horizon_days or self._default_horizon

        logger.info(
            "Projecting costs",
            tenant_id=tenant_id,
            horizon_days=effective_horizon,
            projection_model=projection_model,
        )

        costs = [float(d.get("cost_usd", 0.0)) for d in daily_costs]
        n = len(costs)

        if n < MIN_DATA_POINTS_FOR_TREND:
            mean_fallback = sum(costs) / n if costs else 0.0
            return {
                "tenant_id": tenant_id,
                "horizon_days": effective_horizon,
                "projection_model": "constant_mean",
                "projected_daily_costs": [
                    {
                        "day_offset": i + 1,
                        "projected_cost_usd": round(mean_fallback, 4),
                        "confidence_low": round(mean_fallback * 0.80, 4),
                        "confidence_high": round(mean_fallback * 1.20, 4),
                    }
                    for i in range(effective_horizon)
                ],
                "projected_period_total_usd": round(mean_fallback * effective_horizon, 2),
                "confidence_band_low_usd": round(mean_fallback * 0.80 * effective_horizon, 2),
                "confidence_band_high_usd": round(mean_fallback * 1.20 * effective_horizon, 2),
                "message": "Insufficient data for trend-based projection. Using mean.",
            }

        slope = trend_analysis.get("slope_usd_per_day", 0.0)
        intercept = trend_analysis.get("intercept_usd", costs[-1])
        r_squared = trend_analysis.get("r_squared", 0.0)
        std_dev = trend_analysis.get("std_dev_usd", 0.0)

        # Confidence interval: wider when r_squared is lower
        confidence_multiplier = 1.96 * std_dev * (2.0 - r_squared)

        projected: list[dict[str, Any]] = []
        total_projected = 0.0

        for offset in range(1, effective_horizon + 1):
            x = n + offset

            if projection_model == "exponential" and slope > 0:
                base_cost = intercept * math.exp(slope * offset / n)
            else:
                base_cost = intercept + slope * x

            base_cost = max(0.0, base_cost)
            conf_low = max(0.0, base_cost - confidence_multiplier)
            conf_high = base_cost + confidence_multiplier
            total_projected += base_cost

            proj_date = (
                datetime.now(tz=timezone.utc) + timedelta(days=offset)
            ).date().isoformat()

            projected.append({
                "day_offset": offset,
                "projected_date": proj_date,
                "projected_cost_usd": round(base_cost, 4),
                "confidence_low_usd": round(conf_low, 4),
                "confidence_high_usd": round(conf_high, 4),
            })

        result: dict[str, Any] = {
            "tenant_id": tenant_id,
            "horizon_days": effective_horizon,
            "projection_model": projection_model,
            "projected_daily_costs": projected,
            "projected_period_total_usd": round(total_projected, 2),
            "confidence_band_low_usd": round(
                sum(p["confidence_low_usd"] for p in projected), 2
            ),
            "confidence_band_high_usd": round(
                sum(p["confidence_high_usd"] for p in projected), 2
            ),
            "r_squared": round(r_squared, 4),
            "projected_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "Cost projection completed",
            tenant_id=tenant_id,
            horizon_days=effective_horizon,
            projected_total=result["projected_period_total_usd"],
        )

        return result

    async def detect_seasonal_patterns(
        self,
        tenant_id: str,
        daily_costs: list[dict[str, Any]],
        period_days: int = 7,
    ) -> dict[str, Any]:
        """Detect seasonal/cyclical patterns in cost data.

        Args:
            tenant_id: Tenant identifier.
            daily_costs: Historical daily cost data (needs >= 14 days).
            period_days: Expected seasonality period in days (7 = weekly).

        Returns:
            Dict with has_seasonal_pattern, seasonal_strength, seasonal_indices
            (one per period position), peak_day_of_period, trough_day_of_period,
            and peak_to_trough_ratio fields.
        """
        logger.info(
            "Detecting seasonal patterns",
            tenant_id=tenant_id,
            data_points=len(daily_costs),
            period_days=period_days,
        )

        costs = [float(d.get("cost_usd", 0.0)) for d in daily_costs]

        if len(costs) < MIN_DATA_POINTS_FOR_SEASONAL:
            return {
                "tenant_id": tenant_id,
                "has_seasonal_pattern": False,
                "message": f"At least {MIN_DATA_POINTS_FOR_SEASONAL} days required for seasonal analysis.",
            }

        # Compute seasonal indices: average cost for each day-of-period position
        period_buckets: dict[int, list[float]] = {i: [] for i in range(period_days)}
        for idx, cost in enumerate(costs):
            period_position = idx % period_days
            period_buckets[period_position].append(cost)

        seasonal_indices: dict[int, float] = {}
        overall_mean = sum(costs) / len(costs)

        for position, bucket_costs in period_buckets.items():
            if bucket_costs and overall_mean > 0:
                seasonal_indices[position] = round(
                    sum(bucket_costs) / len(bucket_costs) / overall_mean, 4
                )
            else:
                seasonal_indices[position] = 1.0

        if not seasonal_indices:
            return {
                "tenant_id": tenant_id,
                "has_seasonal_pattern": False,
            }

        max_index = max(seasonal_indices.values())
        min_index = min(seasonal_indices.values())
        seasonal_strength = round(max_index - min_index, 4)

        has_seasonal = seasonal_strength > 0.15
        peak_day = max(seasonal_indices, key=lambda k: seasonal_indices[k])
        trough_day = min(seasonal_indices, key=lambda k: seasonal_indices[k])
        peak_to_trough = round(max_index / min_index, 4) if min_index > 0 else 1.0

        result: dict[str, Any] = {
            "tenant_id": tenant_id,
            "has_seasonal_pattern": has_seasonal,
            "period_days": period_days,
            "seasonal_strength": seasonal_strength,
            "seasonal_indices": seasonal_indices,
            "peak_day_of_period": peak_day,
            "trough_day_of_period": trough_day,
            "peak_to_trough_ratio": peak_to_trough,
            "pattern_description": (
                f"{'Strong' if seasonal_strength > 0.30 else 'Moderate' if seasonal_strength > 0.15 else 'Weak'} "
                f"{'weekly' if period_days == 7 else f'{period_days}-day'} seasonality detected. "
                f"Peak on period day {peak_day}, trough on day {trough_day}."
                if has_seasonal
                else "No significant seasonal pattern detected."
            ),
            "analyzed_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "Seasonal pattern detection completed",
            tenant_id=tenant_id,
            has_seasonal_pattern=has_seasonal,
            seasonal_strength=seasonal_strength,
        )

        return result

    async def compare_budget_vs_actual(
        self,
        tenant_id: str,
        budget_limit_usd: float,
        period_start: datetime,
        period_end: datetime,
        actual_daily_costs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare actual spend vs budget for a period.

        Args:
            tenant_id: Tenant identifier.
            budget_limit_usd: Total budget for the period.
            period_start: Start of the budget period.
            period_end: End of the budget period.
            actual_daily_costs: List of daily cost dicts with date and cost_usd.

        Returns:
            Dict with total_actual_usd, budget_limit_usd, variance_usd,
            utilization_percent, on_track (bool), daily_budget_usd,
            days_remaining, and status_message fields.
        """
        logger.info(
            "Comparing budget vs actual",
            tenant_id=tenant_id,
            budget_limit_usd=budget_limit_usd,
        )

        total_actual = sum(float(d.get("cost_usd", 0.0)) for d in actual_daily_costs)
        period_days = max(1, (period_end - period_start).days)
        elapsed_days = (datetime.now(tz=timezone.utc) - period_start).days
        remaining_days = max(0, period_days - elapsed_days)

        utilization = total_actual / budget_limit_usd if budget_limit_usd > 0 else 0.0
        expected_utilization = elapsed_days / period_days if period_days > 0 else 0.0
        variance_usd = total_actual - (budget_limit_usd * expected_utilization)

        daily_budget = budget_limit_usd / period_days if period_days > 0 else 0.0
        on_track = utilization <= expected_utilization * 1.10

        if utilization >= 1.0:
            status = "over_budget"
        elif utilization >= 0.90:
            status = "near_limit"
        elif on_track:
            status = "on_track"
        else:
            status = "at_risk"

        result: dict[str, Any] = {
            "tenant_id": tenant_id,
            "budget_limit_usd": budget_limit_usd,
            "total_actual_usd": round(total_actual, 2),
            "variance_usd": round(variance_usd, 2),
            "utilization_percent": round(utilization * 100.0, 2),
            "expected_utilization_percent": round(expected_utilization * 100.0, 2),
            "daily_budget_usd": round(daily_budget, 4),
            "elapsed_days": elapsed_days,
            "days_remaining": remaining_days,
            "on_track": on_track,
            "status": status,
            "status_message": (
                f"{'On track' if on_track else 'Off track'}: "
                f"${total_actual:,.2f} spent of ${budget_limit_usd:,.2f} budget "
                f"({utilization * 100:.1f}% utilization, {elapsed_days}/{period_days} days elapsed)."
            ),
            "analyzed_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "Budget vs actual comparison completed",
            tenant_id=tenant_id,
            utilization_percent=result["utilization_percent"],
            status=status,
        )

        return result

    async def predict_budget_exhaustion(
        self,
        tenant_id: str,
        budget_limit_usd: float,
        amount_spent_usd: float,
        trend_analysis: dict[str, Any],
        period_end: datetime,
    ) -> dict[str, Any]:
        """Predict when the budget will be exhausted given current spend trajectory.

        Args:
            tenant_id: Tenant identifier.
            budget_limit_usd: Total budget for the period.
            amount_spent_usd: Amount spent so far.
            trend_analysis: Output from analyze_cost_trends with slope_usd_per_day.
            period_end: Budget period end date for context.

        Returns:
            Dict with will_exhaust_before_period_end (bool), exhaustion_date (str | None),
            days_until_exhaustion (int | None), days_to_period_end, and
            projected_overspend_usd (float | None) fields.
        """
        logger.info(
            "Predicting budget exhaustion",
            tenant_id=tenant_id,
            budget_limit_usd=budget_limit_usd,
            amount_spent_usd=amount_spent_usd,
        )

        remaining_budget = budget_limit_usd - amount_spent_usd
        daily_slope = trend_analysis.get("slope_usd_per_day", 0.0)
        mean_daily_cost = trend_analysis.get("mean_daily_cost_usd", 0.0)

        now = datetime.now(tz=timezone.utc)
        days_to_period_end = max(0, (period_end - now).days)

        exhaustion_date: str | None = None
        days_until_exhaustion: int | None = None
        will_exhaust: bool = False
        projected_overspend: float | None = None

        if remaining_budget <= 0:
            will_exhaust = True
            exhaustion_date = now.date().isoformat()
            days_until_exhaustion = 0
            projected_overspend = abs(remaining_budget) + mean_daily_cost * days_to_period_end

        elif mean_daily_cost > 0:
            if daily_slope > 0:
                # Quadratic projection: solve for when cumulative spend = remaining_budget
                # cum = mean_daily * t + 0.5 * slope * t^2
                # 0.5 * slope * t^2 + mean_daily * t - remaining = 0
                a = 0.5 * daily_slope
                b = mean_daily_cost
                c = -remaining_budget
                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0 and a > 0:
                    days_to_exhaustion = (-b + math.sqrt(discriminant)) / (2 * a)
                else:
                    days_to_exhaustion = remaining_budget / mean_daily_cost
            else:
                days_to_exhaustion = remaining_budget / mean_daily_cost

            days_until_exhaustion = max(0, int(days_to_exhaustion))
            will_exhaust = days_until_exhaustion <= days_to_period_end

            if will_exhaust:
                exhaustion_date = (now + timedelta(days=days_until_exhaustion)).date().isoformat()
                excess_days = days_to_period_end - days_until_exhaustion
                projected_overspend = round(mean_daily_cost * excess_days, 2) if excess_days > 0 else 0.0

        result: dict[str, Any] = {
            "tenant_id": tenant_id,
            "budget_limit_usd": budget_limit_usd,
            "amount_spent_usd": amount_spent_usd,
            "remaining_budget_usd": round(remaining_budget, 2),
            "will_exhaust_before_period_end": will_exhaust,
            "exhaustion_date": exhaustion_date,
            "days_until_exhaustion": days_until_exhaustion,
            "days_to_period_end": days_to_period_end,
            "projected_overspend_usd": projected_overspend,
            "recommendation": (
                "Budget on track — no action required." if not will_exhaust
                else f"Budget will exhaust on {exhaustion_date} — "
                     f"{'review spend immediately' if days_until_exhaustion <= 7 else 'plan cost reduction measures'}."
            ),
            "predicted_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "Budget exhaustion prediction completed",
            tenant_id=tenant_id,
            will_exhaust=will_exhaust,
            exhaustion_date=exhaustion_date,
        )

        return result

    async def detect_cost_anomalies(
        self,
        tenant_id: str,
        daily_costs: list[dict[str, Any]],
        resource_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Detect cost anomalies in historical spend data using Z-score analysis.

        Args:
            tenant_id: Tenant identifier.
            daily_costs: Historical daily cost data.
            resource_type: Optional resource type label.

        Returns:
            List of anomaly dicts with date, cost_usd, z_score, severity,
            and description fields.
        """
        logger.info(
            "Detecting cost anomalies",
            tenant_id=tenant_id,
            data_points=len(daily_costs),
        )

        if len(daily_costs) < MIN_DATA_POINTS_FOR_TREND:
            return []

        costs = [float(d.get("cost_usd", 0.0)) for d in daily_costs]
        mean_cost = sum(costs) / len(costs)
        variance = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return []

        anomalies: list[dict[str, Any]] = []
        for day_data, cost in zip(daily_costs, costs):
            z_score = (cost - mean_cost) / std_dev
            if abs(z_score) >= self._anomaly_threshold:
                severity = (
                    "critical" if abs(z_score) >= self._anomaly_threshold * 1.5
                    else "high"
                )
                anomalies.append({
                    "date": day_data.get("date"),
                    "cost_usd": round(cost, 4),
                    "mean_cost_usd": round(mean_cost, 4),
                    "deviation_usd": round(cost - mean_cost, 4),
                    "z_score": round(z_score, 4),
                    "severity": severity,
                    "direction": "spike" if z_score > 0 else "drop",
                    "resource_type": resource_type,
                    "description": (
                        f"{'Cost spike' if z_score > 0 else 'Cost drop'} detected: "
                        f"${cost:,.2f} is {abs(z_score):.1f} standard deviations from "
                        f"the mean of ${mean_cost:,.2f}."
                    ),
                })

        logger.info(
            "Cost anomaly detection completed",
            tenant_id=tenant_id,
            anomaly_count=len(anomalies),
        )

        return anomalies

    async def generate_forecast_report(
        self,
        tenant_id: str,
        trend_analysis: dict[str, Any],
        projection: dict[str, Any],
        seasonal_patterns: dict[str, Any],
        budget_comparison: dict[str, Any],
        exhaustion_prediction: dict[str, Any],
        anomalies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble a comprehensive cost forecast report.

        Args:
            tenant_id: Tenant identifier.
            trend_analysis: Output from analyze_cost_trends.
            projection: Output from project_costs.
            seasonal_patterns: Output from detect_seasonal_patterns.
            budget_comparison: Output from compare_budget_vs_actual.
            exhaustion_prediction: Output from predict_budget_exhaustion.
            anomalies: Output from detect_cost_anomalies.

        Returns:
            Structured forecast report dict with executive summary,
            findings, and recommended actions.
        """
        logger.info(
            "Generating forecast report",
            tenant_id=tenant_id,
        )

        findings: list[str] = []
        recommendations: list[str] = []
        risk_level = "low"

        trend_type = trend_analysis.get("trend_type", "stable")
        if trend_type in ("linear_growth", "exponential"):
            slope = trend_analysis.get("slope_usd_per_day", 0.0)
            findings.append(f"Spend is trending upward at ${slope:,.2f}/day.")
            recommendations.append("Review workload scaling — cost growth may indicate inefficiency.")
            risk_level = "medium"

        budget_status = budget_comparison.get("status", "on_track")
        if budget_status in ("over_budget", "near_limit"):
            findings.append(f"Budget status: {budget_status}.")
            recommendations.append("Activate cost controls or request budget amendment.")
            risk_level = "high" if budget_status == "over_budget" else "medium"

        if exhaustion_prediction.get("will_exhaust_before_period_end"):
            days = exhaustion_prediction.get("days_until_exhaustion", 0)
            findings.append(f"Budget exhaustion projected in {days} days.")
            recommendations.append(exhaustion_prediction.get("recommendation", ""))
            risk_level = "critical" if days <= 7 else "high"

        if anomalies:
            critical_anomalies = [a for a in anomalies if a.get("severity") == "critical"]
            findings.append(f"{len(anomalies)} cost anomaly(ies) detected ({len(critical_anomalies)} critical).")
            recommendations.append("Investigate anomalous spend events for waste or misconfigurations.")

        if seasonal_patterns.get("has_seasonal_pattern"):
            peak_day = seasonal_patterns.get("peak_day_of_period")
            findings.append(f"Seasonal cost pattern detected — peak on period day {peak_day}.")
            recommendations.append("Consider pre-scaling capacity before peak days to avoid on-demand cost surges.")

        report: dict[str, Any] = {
            "report_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "overall_risk_level": risk_level,
            "executive_summary": (
                f"Cost analysis for tenant {tenant_id}: {trend_type} trend detected. "
                f"Budget status: {budget_status}. "
                f"{len(findings)} finding(s) identified."
            ),
            "findings": findings,
            "recommendations": recommendations,
            "trend_analysis": trend_analysis,
            "projection": projection,
            "seasonal_patterns": seasonal_patterns,
            "budget_comparison": budget_comparison,
            "exhaustion_prediction": exhaustion_prediction,
            "anomaly_count": len(anomalies),
            "critical_anomaly_count": len([a for a in anomalies if a.get("severity") == "critical"]),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "Forecast report generated",
            tenant_id=tenant_id,
            risk_level=risk_level,
            finding_count=len(findings),
        )

        return report

    @staticmethod
    def _linear_regression(
        x_values: list[int],
        y_values: list[float],
    ) -> tuple[float, float, float]:
        """Compute linear regression coefficients and R-squared.

        Args:
            x_values: Independent variable values.
            y_values: Dependent variable values.

        Returns:
            Tuple of (slope, intercept, r_squared).
        """
        n = len(x_values)
        if n < 2:
            return 0.0, y_values[0] if y_values else 0.0, 0.0

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x ** 2 for x in x_values)

        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return 0.0, sum_y / n, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        y_pred = [slope * x + intercept for x in x_values]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_values, y_pred))

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return slope, intercept, max(0.0, min(1.0, r_squared))
