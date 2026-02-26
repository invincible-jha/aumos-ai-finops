"""HTTP client for the KubeCost cost allocation API.

KubeCost is a commercial Kubernetes cost monitoring solution with
a free tier. This client wraps the KubeCost REST API to fetch
allocation and efficiency data.

API docs: https://docs.kubecost.com/apis/apis-overview
"""

from typing import Any

import httpx

from aumos_common.observability import get_logger

from aumos_ai_finops.settings import Settings

logger = get_logger(__name__)


class KubeCostClient:
    """Async HTTP client for the KubeCost cost allocation API.

    Implements IKubeCostClient interface from core/interfaces.py.
    Uses httpx for async HTTP requests with configurable timeouts.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize KubeCostClient with service settings.

        Args:
            settings: Service settings containing kubecost_base_url and timeout config.
        """
        self._base_url = settings.kubecost_base_url.rstrip("/")
        self._timeout = settings.cost_provider_timeout_seconds
        self._enabled = settings.kubecost_enabled

    async def get_allocation(
        self,
        window: str,
        aggregate: str = "namespace",
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch cost allocation data from KubeCost.

        Args:
            window: Time window, e.g. '24h', '7d', or 'lastweek'.
            aggregate: Aggregation dimension: namespace | pod | deployment | controller
            namespace: Optional namespace filter.

        Returns:
            List of allocation dicts from the KubeCost API response.

        Raises:
            RuntimeError: If KubeCost is not enabled.
            httpx.HTTPError: On HTTP or connection errors.
        """
        if not self._enabled:
            raise RuntimeError("KubeCost integration is disabled (AUMOS_FINOPS_KUBECOST_ENABLED=false)")

        params: dict[str, str] = {
            "window": window,
            "aggregate": aggregate,
            "accumulate": "true",
        }
        if namespace is not None:
            params["filterNamespaces"] = namespace

        url = f"{self._base_url}/model/allocation"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                # KubeCost returns {"code": 200, "data": [{name: allocation, ...}]}
                raw_data: list[dict[str, Any]] = data.get("data", [])
                if not raw_data:
                    return []

                # Flatten the allocation sets
                flat_allocations: list[dict[str, Any]] = []
                for allocation_set in raw_data:
                    for name, allocation in allocation_set.items():
                        allocation["name"] = name
                        # Normalize KubeCost field names to match OpenCost convention
                        if "totalCost" not in allocation and "totalCost" in allocation.get("cpuCost", {}):
                            allocation["totalCost"] = (
                                allocation.get("cpuCost", 0)
                                + allocation.get("gpuCost", 0)
                                + allocation.get("ramCost", 0)
                                + allocation.get("pvCost", 0)
                                + allocation.get("networkCost", 0)
                            )
                        flat_allocations.append(allocation)

                return flat_allocations

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "kubecost_http_error",
                    url=url,
                    status_code=exc.response.status_code,
                    response_text=exc.response.text[:500],
                )
                raise
            except httpx.RequestError as exc:
                logger.error(
                    "kubecost_connection_error",
                    url=url,
                    error=str(exc),
                )
                raise

    async def health_check(self) -> bool:
        """Verify KubeCost API is reachable.

        Returns:
            True if the KubeCost API responds successfully, False otherwise.
        """
        if not self._enabled:
            return False

        url = f"{self._base_url}/healthz"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url)
                return response.status_code == 200
        except httpx.RequestError as exc:
            logger.warning("kubecost_health_check_failed", error=str(exc))
            return False
