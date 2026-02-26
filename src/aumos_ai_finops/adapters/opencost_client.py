"""HTTP client for the OpenCost cost allocation API.

OpenCost is an open-source Kubernetes cost monitoring solution.
This client wraps the OpenCost REST API to fetch cost allocation data
by namespace, workload, or label.

API docs: https://www.opencost.io/docs/integrations/api
"""

from typing import Any

import httpx

from aumos_common.observability import get_logger

from aumos_ai_finops.settings import Settings

logger = get_logger(__name__)


class OpenCostClient:
    """Async HTTP client for the OpenCost cost allocation API.

    Implements IOpenCostClient interface from core/interfaces.py.
    Uses httpx for async HTTP requests with configurable timeouts.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize OpenCostClient with service settings.

        Args:
            settings: Service settings containing opencost_base_url and timeout config.
        """
        self._base_url = settings.opencost_base_url.rstrip("/")
        self._timeout = settings.cost_provider_timeout_seconds
        self._enabled = settings.opencost_enabled

    async def get_allocation(
        self,
        namespace: str | None,
        window: str,
        aggregate: str = "namespace",
    ) -> list[dict[str, Any]]:
        """Fetch cost allocation data from OpenCost.

        Args:
            namespace: Kubernetes namespace to filter (None = all namespaces).
            window: Time window string, e.g. '24h', '7d', or ISO date range.
            aggregate: Aggregation dimension: namespace | pod | deployment | label

        Returns:
            List of allocation dicts from the OpenCost API response.

        Raises:
            RuntimeError: If OpenCost is not enabled.
            httpx.HTTPError: On HTTP or connection errors.
        """
        if not self._enabled:
            raise RuntimeError("OpenCost integration is disabled (AUMOS_FINOPS_OPENCOST_ENABLED=false)")

        params: dict[str, str] = {
            "window": window,
            "aggregate": aggregate,
        }
        if namespace is not None:
            params["filter"] = f"namespace:{namespace}"

        url = f"{self._base_url}/model/allocation"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # OpenCost returns {"code": 200, "data": [...]}
                allocations: list[dict[str, Any]] = data.get("data", [])
                if isinstance(allocations, list) and len(allocations) > 0:
                    # Flatten: data is a list of dicts where each dict is {name: allocation}
                    flat_allocations: list[dict[str, Any]] = []
                    for allocation_set in allocations:
                        for name, allocation in allocation_set.items():
                            allocation["name"] = name
                            flat_allocations.append(allocation)
                    return flat_allocations

                return []

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "opencost_http_error",
                    url=url,
                    status_code=exc.response.status_code,
                    response_text=exc.response.text[:500],
                )
                raise
            except httpx.RequestError as exc:
                logger.error(
                    "opencost_connection_error",
                    url=url,
                    error=str(exc),
                )
                raise

    async def health_check(self) -> bool:
        """Verify OpenCost API is reachable.

        Returns:
            True if the OpenCost API responds successfully, False otherwise.
        """
        if not self._enabled:
            return False

        url = f"{self._base_url}/healthz"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url)
                return response.status_code == 200
        except httpx.RequestError as exc:
            logger.warning("opencost_health_check_failed", error=str(exc))
            return False
