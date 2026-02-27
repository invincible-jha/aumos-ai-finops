"""Route modules for the AumOS AI FinOps API.

Provides the chargeback, budget-limit, and attribution routers as sub-routers
that can be mounted alongside the main finops router.
"""
from aumos_ai_finops.api.routes.attribution import router as attribution_router
from aumos_ai_finops.api.routes.finops import router

__all__ = ["router", "attribution_router"]
