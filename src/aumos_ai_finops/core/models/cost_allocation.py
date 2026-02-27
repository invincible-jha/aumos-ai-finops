"""Chargeback cost allocation and budget limit models.

This module re-exports CostAllocation and BudgetLimit from the canonical
flat models.py file. The classes are defined there to maintain a single
SQLAlchemy metadata registration and follow the project's file structure
convention (core/models.py is the single source of truth for ORM models).

Import from here or from ``aumos_ai_finops.core.models`` — both work.
"""
from __future__ import annotations

# Re-export from the package __init__ which loads the flat models.py.
# Using a late import to avoid circular import at module load time.
from aumos_ai_finops.core.models import BudgetLimit, CostAllocation

__all__ = ["BudgetLimit", "CostAllocation"]
