# aumos-ai-finops

> Per-tenant GPU cost tracking, token analytics, ROI measurement, budget alerting, cost-optimized model routing, and executive investment dashboards.

Part of the [AumOS Enterprise](https://aumos.ai) composable AI platform.

## Why This Exists

Only 23% of enterprises can measure AI ROI. This service provides the financial observability layer that answers:
- How much are we spending on GPU and LLM tokens per tenant?
- What is the measurable ROI of our AI initiatives?
- Which models are most cost-effective for our workloads?
- Are we on track against our AI spending budgets?

## Features

- **Cost Tracking** — GPU, CPU, storage, and network cost allocation via OpenCost/KubeCost integration
- **Token Analytics** — Per-model token consumption and cost with tier-based pricing
- **ROI Engine** — Multi-touch attribution: productivity + quality + risk avoidance vs AI cost
- **Budget Alerts** — Configurable warning/critical thresholds with Kafka events
- **Routing Optimizer** — Weighted scoring across cost, quality, and latency to recommend optimal models
- **Executive Dashboard** — Single-view summary of all FinOps metrics

## Quick Start

```bash
# Install dependencies
make dev

# Copy and configure environment
cp .env.example .env

# Start local infrastructure
make docker-up

# Run database migrations
make migrate-up

# Start the service
make run
```

The API will be available at `http://localhost:8000`. OpenAPI docs at `http://localhost:8000/docs`.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/finops/costs` | Cost breakdown by resource/tenant |
| GET | `/api/v1/finops/costs/gpu` | GPU-specific cost tracking |
| POST | `/api/v1/finops/costs/record` | Manually record a cost entry |
| POST | `/api/v1/finops/costs/sync/opencost` | Sync costs from OpenCost |
| GET | `/api/v1/finops/tokens` | Token consumption analytics |
| GET | `/api/v1/finops/tokens/by-model` | Per-model token usage |
| POST | `/api/v1/finops/tokens/record` | Record token usage |
| POST | `/api/v1/finops/roi/calculate` | Calculate ROI for an AI initiative |
| GET | `/api/v1/finops/roi/reports` | List ROI reports |
| POST | `/api/v1/finops/budgets` | Create budget threshold |
| GET | `/api/v1/finops/budgets` | List budgets |
| GET | `/api/v1/finops/budgets/{id}/alerts` | Budget alerts |
| POST | `/api/v1/finops/budgets/evaluate` | Evaluate and trigger alerts |
| GET | `/api/v1/finops/dashboard` | Executive dashboard data |
| POST | `/api/v1/finops/routing/optimize` | Cost-optimized routing recommendation |

## ROI Formula

```
ROI = (productivity_gain + quality_improvement + risk_avoidance - ai_cost) / ai_cost × 100

Where:
  productivity_gain    = hours_saved × hourly_rate × headcount
  quality_improvement  = error_reduction_rate × avg_error_cost
  risk_avoidance       = incidents_prevented × avg_incident_cost
  ai_cost              = gpu_cost + token_cost + infra_cost
```

## Architecture

```
src/aumos_ai_finops/
├── api/          # FastAPI routes + Pydantic schemas
├── core/         # Business logic services + interfaces + ORM models
└── adapters/     # Repositories + Kafka publisher + OpenCost/KubeCost clients
```

All services use constructor injection against Protocol interfaces. OpenCost and KubeCost
clients are optional — disable with `AUMOS_FINOPS_OPENCOST_ENABLED=false`.

## Kafka Events

| Topic | Trigger |
|-------|---------|
| `finops.cost_recorded` | New cost record persisted |
| `finops.budget_exceeded` | Budget threshold breached |
| `finops.roi_calculated` | ROI calculation completed |

## Database Tables

All tables use the `fin_` prefix with PostgreSQL Row-Level Security for tenant isolation.

| Table | Purpose |
|-------|---------|
| `fin_cost_records` | GPU/CPU/storage cost allocations |
| `fin_token_usage` | Per-model token consumption |
| `fin_roi_calculations` | Multi-touch attribution ROI |
| `fin_budgets` | Per-tenant budget thresholds |
| `fin_budget_alerts` | Triggered threshold alerts |
| `fin_routing_recommendations` | Cost-optimized routing suggestions |

## Development

```bash
make lint         # Ruff linting
make format       # Ruff formatting
make type-check   # mypy strict mode
make test         # pytest with coverage
make check        # Full quality gate
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
