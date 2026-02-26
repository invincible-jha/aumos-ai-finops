# CLAUDE.md — AumOS AI FinOps

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-ai-finops`) is part of **Tier A: Platform Services**:
Financial operations, cost intelligence, and ROI measurement for AI workloads.

**Release Tier:** A: Fully Open
**Product Mapping:** Product 7 — AI FinOps & ROI Intelligence
**Phase:** 3 (Months 9-12)

## Repo Purpose

Provides per-tenant GPU cost tracking, token consumption analytics, ROI measurement,
budget threshold alerting, cost-optimized model routing recommendations, and executive
investment dashboards. Addresses the fact that only 23% of enterprises can measure AI ROI.
Integrates with OpenCost and KubeCost for Kubernetes cost data, and publishes events to
downstream analytics pipelines.

## Architecture Position

```
aumos-platform-core → aumos-auth-gateway → THIS REPO → aumos-observability
aumos-llm-serving   ────────────────────────────────↗ (token data consumer)
aumos-model-registry ───────────────────────────────↗ (model cost metadata)
                                         ↘ aumos-event-bus (publishes events)
                                         ↘ aumos-data-layer (stores cost records)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-llm-serving` — Token usage metadata via Kafka events
- `aumos-model-registry` — Model cost tier information

**Downstream dependents (other repos IMPORT from this):**
- `aumos-governance-engine` — Budget compliance signals
- `aumos-observability` — Cost metrics for unified dashboard

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| httpx | 0.27+ | Async HTTP client for OpenCost/KubeCost APIs |
| python-dateutil | 2.9+ | Date range calculations for ROI periods |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.**
   ```python
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**
   - `finops.cost_recorded` — after a cost record is persisted
   - `finops.budget_exceeded` — when a budget threshold is breached
   - `finops.roi_calculated` — after an ROI calculation completes

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### File Structure Convention

```
src/aumos_ai_finops/
├── __init__.py
├── main.py                   # FastAPI app entry point
├── settings.py               # Extends AumOSSettings
├── api/
│   ├── __init__.py
│   ├── router.py             # All finops endpoints
│   └── schemas.py            # Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── models.py             # SQLAlchemy ORM models (fin_ prefix)
│   ├── services.py           # CostCollectorService, TokenTrackerService,
│   │                         #   ROIEngineService, BudgetAlertService,
│   │                         #   RoutingOptimizerService
│   └── interfaces.py         # Abstract interfaces (Protocol classes)
├── adapters/
│   ├── __init__.py
│   ├── repositories.py       # SQLAlchemy repositories for all fin_ tables
│   ├── kafka.py              # finops event publishers
│   ├── opencost_client.py    # OpenCost HTTP API client
│   └── kubecost_client.py    # KubeCost HTTP API client
└── migrations/
    ├── env.py
    ├── alembic.ini
    └── versions/
tests/
├── __init__.py
├── conftest.py
├── test_api.py
├── test_services.py
└── test_repositories.py
```

## Domain Concepts

- **Cost Record** (`fin_cost_records`) — GPU/CPU/storage cost allocation per tenant per time window
- **Token Usage** (`fin_token_usage`) — per-model, per-tenant prompt/completion token consumption
- **ROI Calculation** (`fin_roi_calculations`) — multi-touch attribution: productivity + quality + risk_avoidance
- **Budget** (`fin_budgets`) — per-tenant spending threshold with alerting configuration
- **Budget Alert** (`fin_budget_alerts`) — triggered when actual spend approaches/exceeds budget
- **Routing Recommendation** (`fin_routing_recommendations`) — cost-optimized model selection guidance

## ROI Calculation Formula

```
ROI = (productivity_gain_usd + quality_improvement_usd + risk_avoidance_usd - ai_cost_usd)
      / ai_cost_usd * 100

Components:
  productivity_gain_usd  = hours_saved × hourly_rate × headcount
  quality_improvement_usd = error_reduction_rate × avg_error_cost
  risk_avoidance_usd     = incidents_prevented × avg_incident_cost
  ai_cost_usd            = gpu_cost + token_cost + infra_cost
```

## Environment Variables

Standard AumOS vars from `aumos_common.config.AumOSSettings`.
Repo-specific vars use prefix `AUMOS_FINOPS_`:

```
AUMOS_FINOPS_OPENCOST_BASE_URL=http://opencost-service:9090
AUMOS_FINOPS_KUBECOST_BASE_URL=http://kubecost-service:9090
AUMOS_FINOPS_OPENCOST_ENABLED=true
AUMOS_FINOPS_KUBECOST_ENABLED=false
AUMOS_FINOPS_DEFAULT_BUDGET_ALERT_THRESHOLD=0.80
AUMOS_FINOPS_ROI_LOOKBACK_DAYS=30
AUMOS_FINOPS_GPU_COST_PER_HOUR_A100=2.21
AUMOS_FINOPS_GPU_COST_PER_HOUR_H100=4.76
AUMOS_FINOPS_GPU_COST_PER_HOUR_T4=0.35
```

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.**
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode GPU costs or thresholds.** All configurable via settings.
6. **Do NOT skip type hints.** Every function signature must be typed.
7. **Do NOT put business logic in API routes.**
8. **Do NOT bypass RLS** for multi-tenant cost data.
9. **Do NOT call OpenCost/KubeCost synchronously** — always use async httpx client.
10. **Do NOT store PII** in cost records — use tenant_id and model_id, not user emails.
