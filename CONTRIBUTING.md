# Contributing to aumos-ai-finops

## Development Setup

```bash
git clone <repo-url>
cd aumos-ai-finops
make dev
cp .env.example .env
make docker-up
make migrate-up
```

## Code Standards

- Python 3.11+ with strict type hints on all function signatures
- `ruff` for linting and formatting (`make format && make lint`)
- `mypy` strict mode (`make type-check`)
- Minimum 80% test coverage on core modules

## Branching

- Feature branches: `feature/<description>`
- Bug fixes: `fix/<description>`
- Docs: `docs/<description>`
- Branch from `main`, squash-merge PRs

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add KubeCost GPU allocation sync
fix: correct tier2 completion token pricing factor
refactor: extract ROI calculation into standalone method
test: add budget evaluation threshold boundary tests
docs: update ROI formula documentation
```

## Adding a New Endpoint

1. Add Pydantic schemas to `src/aumos_ai_finops/api/schemas.py`
2. Add service method to the appropriate service in `src/aumos_ai_finops/core/services.py`
3. Add route handler to `src/aumos_ai_finops/api/router.py`
4. Add tests to `tests/test_api.py` and `tests/test_services.py`

## Architecture Rules

- No business logic in API routes — delegate entirely to services
- Services depend on interfaces, not concrete repositories
- All external I/O is async
- Publish Kafka events after all state-changing operations
- RLS enforced via `get_db_session` — never bypass
