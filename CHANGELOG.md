# Changelog

All notable changes to `aumos-ai-finops` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-02-26

### Added
- Initial implementation of AumOS AI FinOps service
- `CostCollectorService` — GPU/CPU/storage cost tracking with OpenCost and KubeCost integration
- `TokenTrackerService` — Per-model token consumption analytics with tier-based pricing
- `ROIEngineService` — Multi-touch attribution ROI calculation (productivity + quality + risk avoidance)
- `BudgetAlertService` — Per-tenant budget thresholds with warning/critical alert evaluation
- `RoutingOptimizerService` — Weighted multi-objective model routing recommendation
- Six database tables with `fin_` prefix and PostgreSQL RLS tenant isolation
- Kafka events: `finops.cost_recorded`, `finops.budget_exceeded`, `finops.roi_calculated`
- OpenCost HTTP client for Kubernetes cost allocation sync
- KubeCost HTTP client with field normalization
- Executive dashboard endpoint aggregating all FinOps metrics
- Full API surface: 15 endpoints across costs, tokens, ROI, budgets, routing, and dashboard
- Alembic migration for initial schema
- Docker and docker-compose.dev.yml for local development
