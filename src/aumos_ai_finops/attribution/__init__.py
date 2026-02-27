"""Cost-to-outcome attribution for AI decision ROI measurement.

This package joins AI inference costs with business outcome events to compute
ROI per decision. The core value proposition: show exactly how much each AI
decision cost and whether it was worth it.

Modules:
    cost_outcome_attributor — CostOutcomeAttributor: collects costs and joins outcomes
    outcome_adapters.webhook — WebhookOutcomeAdapter: receives outcomes via HTTP webhook
    outcome_adapters.manual — ManualEntryOutcomeAdapter: manual outcome entry
"""

from aumos_ai_finops.attribution.cost_outcome_attributor import CostOutcomeAttributor

__all__ = ["CostOutcomeAttributor"]
