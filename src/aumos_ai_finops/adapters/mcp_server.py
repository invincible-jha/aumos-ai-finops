"""MCP server exposing AumOS FinOps cost data to AI agents.

Implements a Model Context Protocol (MCP) server that gives AI agents
structured access to AI spending data, model cost efficiency reports,
budget status, and cost optimisation recommendations.

Gap Coverage: GAP-267 (MCP Server for AI Agent Queries)

Tools exposed via MCP:
    get_current_month_spend       — Total AI spend for the current calendar month
    get_model_cost_efficiency     — Cost efficiency comparison across models
    get_budget_status             — Current budget utilisation for a tenant/team
    get_cost_recommendations      — AI-generated cost optimisation recommendations

Usage:
    The server can be embedded in the FastAPI lifespan or run as a standalone
    process via ``uvicorn`` on a separate port. Most FinOps use cases attach it
    to a stdio transport for Claude Desktop / agent framework integration.

    from aumos_ai_finops.adapters.mcp_server import FinOpsMcpServer

    server = FinOpsMcpServer(cost_service=..., roi_service=...)
    await server.run_stdio()   # standalone agent process
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Minimal type stubs — avoids hard MCP import at module load time so the
# rest of the service can still start if mcp is not installed.
# The FinOpsMcpServer.run_*() methods will raise ImportError clearly.
# ---------------------------------------------------------------------------


def _import_mcp() -> Any:
    """Import the mcp package lazily and raise a clear error if absent."""
    try:
        import mcp  # noqa: PLC0415

        return mcp
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for GAP-267. "
            "Add 'mcp>=1.0' to your dependencies and reinstall."
        ) from exc


# ---------------------------------------------------------------------------
# FinOpsMcpServer
# ---------------------------------------------------------------------------


class FinOpsMcpServer:
    """MCP server exposing AumOS FinOps data to AI agents.

    Allows AI agents (Claude, GPT, Gemini, etc.) to query cost data,
    compare model efficiency, check budget status, and receive cost
    optimisation recommendations in natural language via MCP tools.

    The server wraps the existing CostCollectorService and ROIEngineService
    — it does NOT bypass the service layer or talk directly to the database.

    Args:
        cost_service: Injected CostCollectorService for cost data queries.
        roi_service: Injected ROIEngineService for efficiency calculations.
        server_name: MCP server name advertised to connecting clients.
    """

    def __init__(
        self,
        cost_service: Any,
        roi_service: Any,
        server_name: str = "aumos-finops",
    ) -> None:
        """Initialise the MCP server with injected services.

        Args:
            cost_service: CostCollectorService instance.
            roi_service: ROIEngineService instance.
            server_name: Name exposed to MCP clients.
        """
        self._cost_service = cost_service
        self._roi_service = roi_service
        self._server_name = server_name
        self._server: Any = None  # Lazily initialised on first run

    # ------------------------------------------------------------------
    # Server bootstrap
    # ------------------------------------------------------------------

    def _build_server(self) -> Any:
        """Build and configure the MCP server instance.

        Returns:
            Configured mcp.Server instance with all tools registered.
        """
        mcp = _import_mcp()
        server = mcp.Server(self._server_name)
        self._register_tools(server)
        logger.info("FinOpsMcpServer configured", server_name=self._server_name)
        return server

    async def run_stdio(self) -> None:
        """Run the MCP server on stdio transport.

        Intended for standalone subprocess use with Claude Desktop or
        other MCP host applications that manage server processes.
        """
        mcp = _import_mcp()
        if self._server is None:
            self._server = self._build_server()
        async with mcp.stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_tools(self, server: Any) -> None:
        """Register all MCP tools on the server.

        Args:
            server: mcp.Server instance to register tools on.
        """
        mcp = _import_mcp()

        @server.list_tools()
        async def list_tools() -> list[Any]:
            """Advertise available FinOps tools to MCP clients."""
            return [
                mcp.types.Tool(
                    name="get_current_month_spend",
                    description=(
                        "Get the total AI infrastructure and LLM API spending for the "
                        "current calendar month for a given tenant. Returns spend broken "
                        "down by GPU costs, token costs, and external LLM API costs."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tenant_id": {
                                "type": "string",
                                "description": "UUID of the tenant to query.",
                            },
                        },
                        "required": ["tenant_id"],
                    },
                ),
                mcp.types.Tool(
                    name="get_model_cost_efficiency",
                    description=(
                        "Compare cost efficiency (cost per successful task) across "
                        "all AI models used by a tenant. Optionally filter by task type. "
                        "Returns models ranked by cost efficiency."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tenant_id": {
                                "type": "string",
                                "description": "UUID of the tenant to query.",
                            },
                            "task_type": {
                                "type": "string",
                                "description": (
                                    "Optional task type filter (e.g., 'classification', "
                                    "'summarisation', 'code_review'). Omit to compare "
                                    "across all task types."
                                ),
                            },
                        },
                        "required": ["tenant_id"],
                    },
                ),
                mcp.types.Tool(
                    name="get_budget_status",
                    description=(
                        "Check the current budget utilisation for a tenant or specific team. "
                        "Returns consumed spend, remaining budget, utilisation percentage, "
                        "and a projected overrun warning if the current burn rate continues."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tenant_id": {
                                "type": "string",
                                "description": "UUID of the tenant to query.",
                            },
                            "team_id": {
                                "type": "string",
                                "description": (
                                    "Optional team identifier. Omit for tenant-wide status."
                                ),
                            },
                        },
                        "required": ["tenant_id"],
                    },
                ),
                mcp.types.Tool(
                    name="get_cost_recommendations",
                    description=(
                        "Get AI-generated cost optimisation recommendations for a tenant. "
                        "Analyses spending patterns and returns prioritised suggestions "
                        "such as model downgrades, caching opportunities, and underutilised "
                        "reservations."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tenant_id": {
                                "type": "string",
                                "description": "UUID of the tenant to query.",
                            },
                            "max_recommendations": {
                                "type": "integer",
                                "description": "Maximum number of recommendations to return (default 5).",
                                "minimum": 1,
                                "maximum": 20,
                            },
                        },
                        "required": ["tenant_id"],
                    },
                ),
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            """Dispatch an MCP tool call to the appropriate handler.

            Args:
                name: Tool name from the MCP request.
                arguments: Tool arguments from the MCP client.

            Returns:
                List of mcp.types.TextContent with the tool result.
            """
            try:
                if name == "get_current_month_spend":
                    result = await self._handle_current_month_spend(arguments)
                elif name == "get_model_cost_efficiency":
                    result = await self._handle_model_cost_efficiency(arguments)
                elif name == "get_budget_status":
                    result = await self._handle_budget_status(arguments)
                elif name == "get_cost_recommendations":
                    result = await self._handle_cost_recommendations(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                import json  # noqa: PLC0415

                return [mcp.types.TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as exc:
                logger.error("MCP tool call failed", tool=name, error=str(exc))
                return [
                    mcp.types.TextContent(
                        type="text",
                        text=f'{{"error": "Tool execution failed: {exc}"}}',
                    )
                ]

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _handle_current_month_spend(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle the get_current_month_spend tool call.

        Args:
            arguments: Tool arguments containing tenant_id.

        Returns:
            Dict with current month spend breakdown.
        """
        tenant_id_str = arguments.get("tenant_id", "")
        try:
            tenant_id = uuid.UUID(tenant_id_str)
        except ValueError:
            return {"error": f"Invalid tenant_id: {tenant_id_str}"}

        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = now

        try:
            # Delegate to the existing CostCollectorService
            summary = await self._cost_service.get_cost_summary(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=period_end,
            )
            return {
                "tenant_id": tenant_id_str,
                "period": {
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat(),
                },
                "spend": summary,
                "currency": "USD",
            }
        except Exception as exc:
            logger.warning(
                "get_current_month_spend failed — returning stub",
                tenant_id=tenant_id_str,
                error=str(exc),
            )
            # Return structured stub so the agent still gets a useful response
            return {
                "tenant_id": tenant_id_str,
                "period": {
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat(),
                },
                "spend": {
                    "total_usd": 0.0,
                    "gpu_cost_usd": 0.0,
                    "token_cost_usd": 0.0,
                    "external_llm_cost_usd": 0.0,
                },
                "currency": "USD",
                "note": "Cost data unavailable — service may not be initialised.",
            }

    async def _handle_model_cost_efficiency(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle the get_model_cost_efficiency tool call.

        Args:
            arguments: Tool arguments with tenant_id and optional task_type.

        Returns:
            Dict with models ranked by cost efficiency.
        """
        tenant_id_str = arguments.get("tenant_id", "")
        task_type: str | None = arguments.get("task_type")

        try:
            tenant_id = uuid.UUID(tenant_id_str)
        except ValueError:
            return {"error": f"Invalid tenant_id: {tenant_id_str}"}

        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        try:
            efficiency_data = await self._roi_service.get_model_efficiency_ranking(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=now,
                task_type=task_type,
            )
            return {
                "tenant_id": tenant_id_str,
                "task_type_filter": task_type,
                "models": efficiency_data,
            }
        except Exception as exc:
            logger.warning(
                "get_model_cost_efficiency failed — returning stub",
                tenant_id=tenant_id_str,
                error=str(exc),
            )
            return {
                "tenant_id": tenant_id_str,
                "task_type_filter": task_type,
                "models": [],
                "note": "Efficiency data unavailable — ROI service may not be initialised.",
            }

    async def _handle_budget_status(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle the get_budget_status tool call.

        Args:
            arguments: Tool arguments with tenant_id and optional team_id.

        Returns:
            Dict with budget utilisation details.
        """
        tenant_id_str = arguments.get("tenant_id", "")
        team_id: str | None = arguments.get("team_id")

        try:
            tenant_id = uuid.UUID(tenant_id_str)
        except ValueError:
            return {"error": f"Invalid tenant_id: {tenant_id_str}"}

        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        try:
            status = await self._cost_service.get_budget_utilisation(
                tenant_id=tenant_id,
                team_id=team_id,
                period_start=period_start,
                period_end=now,
            )
            return {
                "tenant_id": tenant_id_str,
                "team_id": team_id,
                "budget_status": status,
            }
        except Exception as exc:
            logger.warning(
                "get_budget_status failed — returning stub",
                tenant_id=tenant_id_str,
                team_id=team_id,
                error=str(exc),
            )
            return {
                "tenant_id": tenant_id_str,
                "team_id": team_id,
                "budget_status": {
                    "limit_usd": None,
                    "consumed_usd": 0.0,
                    "remaining_usd": None,
                    "utilization_pct": 0.0,
                    "alert_triggered": False,
                },
                "note": "Budget data unavailable — budget service may not be initialised.",
            }

    async def _handle_cost_recommendations(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle the get_cost_recommendations tool call.

        Generates prioritised cost optimisation recommendations by analysing
        spending patterns and model utilisation for the current month.

        Args:
            arguments: Tool arguments with tenant_id and max_recommendations.

        Returns:
            Dict with list of recommendations sorted by estimated savings.
        """
        tenant_id_str = arguments.get("tenant_id", "")
        max_recommendations: int = int(arguments.get("max_recommendations", 5))

        try:
            tenant_id = uuid.UUID(tenant_id_str)
        except ValueError:
            return {"error": f"Invalid tenant_id: {tenant_id_str}"}

        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        try:
            recommendations = await self._roi_service.generate_cost_recommendations(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=now,
                max_recommendations=max_recommendations,
            )
            return {
                "tenant_id": tenant_id_str,
                "recommendations": recommendations,
                "generated_at": now.isoformat(),
            }
        except Exception as exc:
            logger.warning(
                "get_cost_recommendations failed — returning static recommendations",
                tenant_id=tenant_id_str,
                error=str(exc),
            )
            # Return static template recommendations as a useful fallback
            return {
                "tenant_id": tenant_id_str,
                "recommendations": [
                    {
                        "priority": 1,
                        "category": "model_downgrade",
                        "title": "Evaluate smaller models for low-complexity tasks",
                        "description": (
                            "Tasks with confidence >= 0.95 may produce equivalent quality "
                            "output with a smaller, cheaper model. Review routing thresholds."
                        ),
                        "estimated_saving_usd_per_month": None,
                        "action": "review_routing_thresholds",
                    },
                    {
                        "priority": 2,
                        "category": "caching",
                        "title": "Enable semantic caching for repeated prompts",
                        "description": (
                            "High request volume with repeated prompt patterns suggests "
                            "semantic caching could reduce token costs by 20-40%."
                        ),
                        "estimated_saving_usd_per_month": None,
                        "action": "enable_semantic_cache",
                    },
                ],
                "generated_at": now.isoformat(),
                "note": "Recommendations are template-based — cost data unavailable.",
            }


__all__ = ["FinOpsMcpServer"]
