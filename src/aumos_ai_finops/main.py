"""AumOS AI FinOps service entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_ai_finops.settings import Settings

logger = get_logger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: object) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    logger.info(
        "aumos-ai-finops starting",
        service=settings.service_name,
        opencost_enabled=settings.opencost_enabled,
        kubecost_enabled=settings.kubecost_enabled,
    )
    init_database(settings.database)
    # Kafka publisher is initialized per-request via dependency injection
    yield
    logger.info("aumos-ai-finops shutting down")


app = create_app(
    service_name="aumos-ai-finops",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=None),  # type: ignore[arg-type]
    ],
)

from aumos_ai_finops.api.router import router  # noqa: E402

app.include_router(router, prefix="/api/v1")
