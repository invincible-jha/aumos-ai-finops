.PHONY: install dev lint format type-check test test-cov migrate-up migrate-down docker-build docker-up docker-down clean

# Install production dependencies
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"

# Lint with ruff
lint:
	ruff check src/ tests/

# Format with ruff
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Type-check with mypy
type-check:
	mypy src/

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage report
test-cov:
	pytest tests/ -v --cov=aumos_ai_finops --cov-report=term-missing --cov-report=html

# Run database migrations
migrate-up:
	cd src/aumos_ai_finops && alembic -c migrations/alembic.ini upgrade head

# Rollback last migration
migrate-down:
	cd src/aumos_ai_finops && alembic -c migrations/alembic.ini downgrade -1

# Generate a new migration
migrate-new:
	cd src/aumos_ai_finops && alembic -c migrations/alembic.ini revision --autogenerate -m "$(MSG)"

# Build Docker image
docker-build:
	docker build -t aumos-ai-finops:latest .

# Start local dev stack
docker-up:
	docker compose -f docker-compose.dev.yml up -d

# Stop local dev stack
docker-down:
	docker compose -f docker-compose.dev.yml down

# Run the service locally
run:
	uvicorn aumos_ai_finops.main:app --host 0.0.0.0 --port 8000 --reload

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/

# Full quality gate (CI equivalent)
check: format lint type-check test
