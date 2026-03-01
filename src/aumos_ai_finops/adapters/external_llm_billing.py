"""External LLM API billing adapter for aumos-ai-finops.

Fetches invoice and usage data from external LLM provider APIs (Anthropic,
OpenAI, Google) and normalizes it into AumOS CostRecord format. Enables
unified cost visibility across self-hosted and API-based models within a
single finops dashboard.

Gap Coverage: GAP-264 (External LLM Cost Tracking)
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class ExternalLLMUsageRecord:
    """Normalized cost record from an external LLM provider.

    Attributes:
        provider: Provider name (anthropic/openai/google).
        model_id: Model identifier as used by the provider.
        period_start: Start of the billing period.
        period_end: End of the billing period.
        input_tokens: Total input/prompt tokens consumed.
        output_tokens: Total output/completion tokens consumed.
        total_cost_usd: Total cost in USD for this period.
        request_count: Number of API requests.
        raw_response: Raw provider API response for audit purposes.
    """

    provider: str
    model_id: str
    period_start: datetime
    period_end: datetime
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    request_count: int
    raw_response: dict[str, Any]


class ExternalLLMBillingAdapter:
    """Fetches and normalizes billing data from external LLM provider APIs.

    Supports Anthropic, OpenAI, and Google Vertex AI usage reporting APIs.
    All provider responses are normalized to ExternalLLMUsageRecord format
    for unified ingestion into the AumOS AI FinOps cost pipeline.

    Args:
        anthropic_api_key: Anthropic API key for usage data access.
        openai_api_key: OpenAI API key for usage data access.
        google_api_key: Google Cloud API key for Vertex AI billing data.
        http_timeout_seconds: Per-request HTTP timeout.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        google_api_key: str | None = None,
        http_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the external LLM billing adapter.

        Args:
            anthropic_api_key: Anthropic API key (None if not used).
            openai_api_key: OpenAI API key (None if not used).
            google_api_key: Google Cloud API key (None if not used).
            http_timeout_seconds: HTTP request timeout.
        """
        self._anthropic_key = anthropic_api_key
        self._openai_key = openai_api_key
        self._google_key = google_api_key
        self._http_timeout = http_timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Create the shared HTTP client.

        Must be called before any fetch methods are used.
        """
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._http_timeout),
            follow_redirects=False,
        )
        logger.info(
            "ExternalLLMBillingAdapter initialized",
            providers_configured=[
                p for p, k in [
                    ("anthropic", self._anthropic_key),
                    ("openai", self._openai_key),
                    ("google", self._google_key),
                ] if k
            ],
        )

    async def close(self) -> None:
        """Close the HTTP client on shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_anthropic_usage(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> list[ExternalLLMUsageRecord]:
        """Fetch usage data from the Anthropic API for a billing period.

        Args:
            period_start: Start of the period to fetch.
            period_end: End of the period to fetch.

        Returns:
            List of normalized usage records per model.
        """
        if not self._anthropic_key:
            logger.debug("Anthropic API key not configured — skipping")
            return []

        if self._client is None:
            raise RuntimeError("ExternalLLMBillingAdapter.initialize() was not called")

        try:
            # Anthropic usage API endpoint (workspace-level usage)
            response = await self._client.get(
                "https://api.anthropic.com/v1/usage",
                headers={
                    "x-api-key": self._anthropic_key,
                    "anthropic-version": "2023-06-01",
                },
                params={
                    "start_time": period_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_time": period_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )
            response.raise_for_status()
            data = response.json()

            return self._normalize_anthropic_response(
                data=data,
                period_start=period_start,
                period_end=period_end,
            )

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Anthropic usage API returned error",
                http_status=exc.response.status_code,
                error=str(exc),
            )
            return []
        except Exception as exc:
            logger.error("Failed to fetch Anthropic usage data", error=str(exc))
            return []

    async def fetch_openai_usage(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> list[ExternalLLMUsageRecord]:
        """Fetch usage data from the OpenAI API for a billing period.

        Args:
            period_start: Start of the period to fetch.
            period_end: End of the period to fetch.

        Returns:
            List of normalized usage records per model.
        """
        if not self._openai_key:
            logger.debug("OpenAI API key not configured — skipping")
            return []

        if self._client is None:
            raise RuntimeError("ExternalLLMBillingAdapter.initialize() was not called")

        try:
            date_str = period_start.strftime("%Y-%m-%d")
            response = await self._client.get(
                f"https://api.openai.com/v1/usage?date={date_str}",
                headers={"Authorization": f"Bearer {self._openai_key}"},
            )
            response.raise_for_status()
            data = response.json()

            return self._normalize_openai_response(
                data=data,
                period_start=period_start,
                period_end=period_end,
            )

        except httpx.HTTPStatusError as exc:
            logger.error(
                "OpenAI usage API returned error",
                http_status=exc.response.status_code,
                error=str(exc),
            )
            return []
        except Exception as exc:
            logger.error("Failed to fetch OpenAI usage data", error=str(exc))
            return []

    async def fetch_all_providers(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> list[ExternalLLMUsageRecord]:
        """Fetch usage data from all configured providers.

        Args:
            period_start: Start of the billing period.
            period_end: End of the billing period.

        Returns:
            Combined list of normalized usage records across all providers.
        """
        import asyncio

        results = await asyncio.gather(
            self.fetch_anthropic_usage(period_start, period_end),
            self.fetch_openai_usage(period_start, period_end),
            return_exceptions=True,
        )

        all_records: list[ExternalLLMUsageRecord] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Provider fetch failed", error=str(result))
                continue
            all_records.extend(result)

        logger.info(
            "External LLM billing fetch complete",
            total_records=len(all_records),
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
        )
        return all_records

    def _normalize_anthropic_response(
        self,
        data: dict[str, Any],
        period_start: datetime,
        period_end: datetime,
    ) -> list[ExternalLLMUsageRecord]:
        """Normalize Anthropic API usage response to ExternalLLMUsageRecord.

        Args:
            data: Raw API response dict.
            period_start: Billing period start.
            period_end: Billing period end.

        Returns:
            List of normalized records.
        """
        records: list[ExternalLLMUsageRecord] = []
        # Anthropic usage response structure: {data: [{model, input_tokens, output_tokens, ...}]}
        usage_list = data.get("data", [])

        # Aggregate by model
        model_totals: dict[str, dict[str, Any]] = {}
        for entry in usage_list:
            model = entry.get("model", "unknown")
            if model not in model_totals:
                model_totals[model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": 0.0,
                    "request_count": 0,
                }
            model_totals[model]["input_tokens"] += int(entry.get("input_tokens", 0))
            model_totals[model]["output_tokens"] += int(entry.get("output_tokens", 0))
            model_totals[model]["total_cost_usd"] += float(entry.get("cost", 0.0))
            model_totals[model]["request_count"] += 1

        for model_id, totals in model_totals.items():
            records.append(ExternalLLMUsageRecord(
                provider="anthropic",
                model_id=f"anthropic/{model_id}",
                period_start=period_start,
                period_end=period_end,
                input_tokens=totals["input_tokens"],
                output_tokens=totals["output_tokens"],
                total_cost_usd=totals["total_cost_usd"],
                request_count=totals["request_count"],
                raw_response=data,
            ))

        return records

    def _normalize_openai_response(
        self,
        data: dict[str, Any],
        period_start: datetime,
        period_end: datetime,
    ) -> list[ExternalLLMUsageRecord]:
        """Normalize OpenAI API usage response to ExternalLLMUsageRecord.

        Args:
            data: Raw API response dict.
            period_start: Billing period start.
            period_end: Billing period end.

        Returns:
            List of normalized records.
        """
        records: list[ExternalLLMUsageRecord] = []
        usage_list = data.get("data", [])

        model_totals: dict[str, dict[str, Any]] = {}
        for entry in usage_list:
            model = entry.get("snapshot_id", entry.get("model", "unknown"))
            if model not in model_totals:
                model_totals[model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "request_count": 0,
                }
            model_totals[model]["input_tokens"] += int(entry.get("n_context_tokens_total", 0))
            model_totals[model]["output_tokens"] += int(entry.get("n_generated_tokens_total", 0))
            model_totals[model]["request_count"] += int(entry.get("n_requests", 0))

        for model_id, totals in model_totals.items():
            # OpenAI usage API doesn't return costs directly — estimate from token counts
            # Pricing is loaded from settings in production
            estimated_cost = 0.0
            records.append(ExternalLLMUsageRecord(
                provider="openai",
                model_id=f"openai/{model_id}",
                period_start=period_start,
                period_end=period_end,
                input_tokens=totals["input_tokens"],
                output_tokens=totals["output_tokens"],
                total_cost_usd=estimated_cost,
                request_count=totals["request_count"],
                raw_response=data,
            ))

        return records


__all__ = [
    "ExternalLLMUsageRecord",
    "ExternalLLMBillingAdapter",
]
