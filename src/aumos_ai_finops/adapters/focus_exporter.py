"""FOCUS 1.3 cost data export adapter for aumos-ai-finops.

Exports AI cost records in the FinOps Open Cost and Usage Specification (FOCUS)
1.3 format. FOCUS is the vendor-neutral standard for cloud cost data that enables
cross-provider cost analysis and tooling interoperability.

Gap Coverage: GAP-262 (FOCUS 1.3 Export)

FOCUS 1.3 specification columns implemented:
    BilledCost, EffectiveCost, ListCost, ContractedCost
    BillingAccountId, BillingAccountName
    ResourceId, ResourceName, ResourceType
    ServiceName, ServiceCategory
    ChargeCategory, ChargeDescription
    UsageQuantity, UsageUnit
    BillingPeriodStart, BillingPeriodEnd
    RegionId, AvailabilityZone
    Tags (x_aumos_tenant_id, x_aumos_model_id)
"""

import csv
import io
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# FOCUS 1.3 column headers in specification order
_FOCUS_COLUMNS = [
    "BilledCost",
    "EffectiveCost",
    "ListCost",
    "ContractedCost",
    "BillingCurrency",
    "BillingAccountId",
    "BillingAccountName",
    "SubAccountId",
    "SubAccountName",
    "ResourceId",
    "ResourceName",
    "ResourceType",
    "ServiceName",
    "ServiceCategory",
    "ChargeCategory",
    "ChargeClass",
    "ChargeDescription",
    "ChargeFrequency",
    "UsageQuantity",
    "UsageUnit",
    "BillingPeriodStart",
    "BillingPeriodEnd",
    "ChargePeriodStart",
    "ChargePeriodEnd",
    "RegionId",
    "RegionName",
    "AvailabilityZone",
    "Tags",
]


@dataclass
class FocusRow:
    """A single FOCUS 1.3 cost data row.

    Attributes:
        billed_cost: The cost charged to the billing account in billing currency.
        effective_cost: The amortized cost after discounts and reservations.
        list_cost: The undiscounted list price cost.
        contracted_cost: The cost at contracted/negotiated rates.
        billing_currency: ISO 4217 currency code (default USD).
        billing_account_id: AumOS tenant UUID as the billing account ID.
        billing_account_name: Human-readable account name.
        sub_account_id: Sub-account identifier (service/workload).
        sub_account_name: Sub-account name.
        resource_id: Unique resource identifier (GPU ID, model endpoint, etc.).
        resource_name: Human-readable resource name.
        resource_type: Resource type (GPU, LLM, Storage, etc.).
        service_name: Service producing the cost (aumos-llm-serving, etc.).
        service_category: FOCUS service category (AI and Machine Learning, etc.).
        charge_category: Usage | Adjustment | Tax | Purchase.
        charge_class: Regular | Correction.
        charge_description: Human-readable description of the charge.
        charge_frequency: Recurring | One-Time | Usage-Based.
        usage_quantity: Quantity consumed in usage_unit.
        usage_unit: Unit of usage measurement (Tokens, GPU-Hours, GB-Hours, etc.).
        billing_period_start: Start of the billing period.
        billing_period_end: End of the billing period.
        charge_period_start: Start of the specific charge window.
        charge_period_end: End of the specific charge window.
        region_id: Cloud region identifier.
        region_name: Human-readable region name.
        availability_zone: Availability zone within the region.
        tags: Dict of FOCUS-compatible tags for cross-cutting dimensions.
    """

    billed_cost: float
    effective_cost: float
    list_cost: float
    contracted_cost: float
    billing_currency: str
    billing_account_id: str
    billing_account_name: str
    sub_account_id: str
    sub_account_name: str
    resource_id: str
    resource_name: str
    resource_type: str
    service_name: str
    service_category: str
    charge_category: str
    charge_class: str
    charge_description: str
    charge_frequency: str
    usage_quantity: float
    usage_unit: str
    billing_period_start: datetime
    billing_period_end: datetime
    charge_period_start: datetime
    charge_period_end: datetime
    region_id: str
    region_name: str
    availability_zone: str
    tags: dict[str, str] = field(default_factory=dict)


class FocusExporter:
    """Exports AumOS AI cost records in FOCUS 1.3 format.

    Converts internal CostRecord and TokenUsage records to FOCUS-compliant
    rows. Supports CSV and JSON-Lines output formats for compatibility with
    FinOps tooling (Apptio, CloudZero, Finout, Vantage, etc.).

    Args:
        platform_region: Default cloud region identifier for exported records.
        platform_zone: Default availability zone.
        billing_account_name: Human-readable billing account name.
    """

    def __init__(
        self,
        platform_region: str = "us-east-1",
        platform_zone: str = "us-east-1a",
        billing_account_name: str = "AumOS Enterprise",
    ) -> None:
        """Initialize the FOCUS exporter.

        Args:
            platform_region: Default region for cost records.
            platform_zone: Default availability zone.
            billing_account_name: Billing account display name.
        """
        self._platform_region = platform_region
        self._platform_zone = platform_zone
        self._billing_account_name = billing_account_name

    def convert_cost_record(
        self,
        record: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> FocusRow:
        """Convert an internal CostRecord dict to a FOCUS 1.3 row.

        Args:
            record: CostRecord dict with standard AumOS fields.
            tenant_id: Owning tenant UUID.

        Returns:
            FocusRow in FOCUS 1.3 format.
        """
        total_cost = float(record.get("total_cost_usd", 0.0))
        period_start: datetime = record.get("period_start", datetime.utcnow())
        period_end: datetime = record.get("period_end", datetime.utcnow())
        resource_type = record.get("resource_type", "GPU")
        model_id = record.get("model_id", "unknown")

        return FocusRow(
            billed_cost=total_cost,
            effective_cost=total_cost,
            list_cost=total_cost,
            contracted_cost=total_cost,
            billing_currency="USD",
            billing_account_id=str(tenant_id),
            billing_account_name=self._billing_account_name,
            sub_account_id=model_id,
            sub_account_name=f"Model: {model_id}",
            resource_id=record.get("resource_id", f"aumos-{resource_type.lower()}-{str(uuid.uuid4())[:8]}"),
            resource_name=record.get("resource_name", f"{resource_type} Resource"),
            resource_type=resource_type,
            service_name="aumos-llm-serving",
            service_category="AI and Machine Learning",
            charge_category="Usage",
            charge_class="Regular",
            charge_description=f"{resource_type} usage for model {model_id}",
            charge_frequency="Usage-Based",
            usage_quantity=float(record.get("usage_quantity", 0.0)),
            usage_unit=record.get("usage_unit", "GPU-Hours"),
            billing_period_start=period_start,
            billing_period_end=period_end,
            charge_period_start=period_start,
            charge_period_end=period_end,
            region_id=self._platform_region,
            region_name=self._platform_region,
            availability_zone=self._platform_zone,
            tags={
                "x_aumos_tenant_id": str(tenant_id),
                "x_aumos_model_id": model_id,
                "x_aumos_resource_type": resource_type,
            },
        )

    def convert_token_usage(
        self,
        token_usage: dict[str, Any],
        tenant_id: uuid.UUID,
        cost_per_1k_input_tokens: float = 0.003,
        cost_per_1k_output_tokens: float = 0.015,
    ) -> list[FocusRow]:
        """Convert a TokenUsage record to FOCUS 1.3 rows (one per charge type).

        Creates separate input and output token charge rows to enable
        granular per-direction token cost analysis.

        Args:
            token_usage: TokenUsage dict with prompt_tokens and completion_tokens.
            tenant_id: Owning tenant UUID.
            cost_per_1k_input_tokens: Input token cost per 1000 tokens.
            cost_per_1k_output_tokens: Output token cost per 1000 tokens.

        Returns:
            List of FocusRow (2 rows: input tokens + output tokens).
        """
        model_id = token_usage.get("model_id", "unknown")
        prompt_tokens = int(token_usage.get("prompt_tokens", 0))
        completion_tokens = int(token_usage.get("completion_tokens", 0))
        recorded_at: datetime = token_usage.get("recorded_at", datetime.utcnow())

        input_cost = (prompt_tokens / 1000.0) * cost_per_1k_input_tokens
        output_cost = (completion_tokens / 1000.0) * cost_per_1k_output_tokens

        rows = []

        if prompt_tokens > 0:
            rows.append(FocusRow(
                billed_cost=input_cost,
                effective_cost=input_cost,
                list_cost=input_cost,
                contracted_cost=input_cost,
                billing_currency="USD",
                billing_account_id=str(tenant_id),
                billing_account_name=self._billing_account_name,
                sub_account_id=model_id,
                sub_account_name=f"Model: {model_id}",
                resource_id=f"tokens-input-{model_id}",
                resource_name=f"{model_id} Input Tokens",
                resource_type="LLM",
                service_name="aumos-llm-serving",
                service_category="AI and Machine Learning",
                charge_category="Usage",
                charge_class="Regular",
                charge_description=f"Input tokens for {model_id}",
                charge_frequency="Usage-Based",
                usage_quantity=float(prompt_tokens),
                usage_unit="Tokens",
                billing_period_start=recorded_at,
                billing_period_end=recorded_at,
                charge_period_start=recorded_at,
                charge_period_end=recorded_at,
                region_id=self._platform_region,
                region_name=self._platform_region,
                availability_zone=self._platform_zone,
                tags={
                    "x_aumos_tenant_id": str(tenant_id),
                    "x_aumos_model_id": model_id,
                    "x_aumos_token_direction": "input",
                },
            ))

        if completion_tokens > 0:
            rows.append(FocusRow(
                billed_cost=output_cost,
                effective_cost=output_cost,
                list_cost=output_cost,
                contracted_cost=output_cost,
                billing_currency="USD",
                billing_account_id=str(tenant_id),
                billing_account_name=self._billing_account_name,
                sub_account_id=model_id,
                sub_account_name=f"Model: {model_id}",
                resource_id=f"tokens-output-{model_id}",
                resource_name=f"{model_id} Output Tokens",
                resource_type="LLM",
                service_name="aumos-llm-serving",
                service_category="AI and Machine Learning",
                charge_category="Usage",
                charge_class="Regular",
                charge_description=f"Output tokens for {model_id}",
                charge_frequency="Usage-Based",
                usage_quantity=float(completion_tokens),
                usage_unit="Tokens",
                billing_period_start=recorded_at,
                billing_period_end=recorded_at,
                charge_period_start=recorded_at,
                charge_period_end=recorded_at,
                region_id=self._platform_region,
                region_name=self._platform_region,
                availability_zone=self._platform_zone,
                tags={
                    "x_aumos_tenant_id": str(tenant_id),
                    "x_aumos_model_id": model_id,
                    "x_aumos_token_direction": "output",
                },
            ))

        return rows

    def export_to_csv(self, rows: list[FocusRow]) -> str:
        """Serialise FOCUS rows to CSV format.

        Outputs a compliant FOCUS 1.3 CSV with all required columns in
        specification order. Datetimes are formatted as ISO 8601.

        Args:
            rows: List of FocusRow objects to export.

        Returns:
            CSV-formatted string with FOCUS 1.3 headers and data rows.
        """
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_FOCUS_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow(self._row_to_dict(row))

        logger.info("FOCUS export complete", format="csv", row_count=len(rows))
        return output.getvalue()

    def export_to_jsonlines(self, rows: list[FocusRow]) -> str:
        """Serialise FOCUS rows to JSON Lines format.

        JSON Lines (newline-delimited JSON) is supported by most FinOps
        tools and data pipeline ingestion endpoints.

        Args:
            rows: List of FocusRow objects to export.

        Returns:
            JSON Lines string (one JSON object per line).
        """
        lines = [json.dumps(self._row_to_dict(row)) for row in rows]
        logger.info("FOCUS export complete", format="jsonlines", row_count=len(rows))
        return "\n".join(lines)

    def _row_to_dict(self, row: FocusRow) -> dict[str, Any]:
        """Convert a FocusRow to a flat dict matching FOCUS column names.

        Args:
            row: FocusRow dataclass instance.

        Returns:
            Dict with FOCUS 1.3 column names as keys.
        """
        return {
            "BilledCost": round(row.billed_cost, 6),
            "EffectiveCost": round(row.effective_cost, 6),
            "ListCost": round(row.list_cost, 6),
            "ContractedCost": round(row.contracted_cost, 6),
            "BillingCurrency": row.billing_currency,
            "BillingAccountId": row.billing_account_id,
            "BillingAccountName": row.billing_account_name,
            "SubAccountId": row.sub_account_id,
            "SubAccountName": row.sub_account_name,
            "ResourceId": row.resource_id,
            "ResourceName": row.resource_name,
            "ResourceType": row.resource_type,
            "ServiceName": row.service_name,
            "ServiceCategory": row.service_category,
            "ChargeCategory": row.charge_category,
            "ChargeClass": row.charge_class,
            "ChargeDescription": row.charge_description,
            "ChargeFrequency": row.charge_frequency,
            "UsageQuantity": round(row.usage_quantity, 6),
            "UsageUnit": row.usage_unit,
            "BillingPeriodStart": row.billing_period_start.isoformat(),
            "BillingPeriodEnd": row.billing_period_end.isoformat(),
            "ChargePeriodStart": row.charge_period_start.isoformat(),
            "ChargePeriodEnd": row.charge_period_end.isoformat(),
            "RegionId": row.region_id,
            "RegionName": row.region_name,
            "AvailabilityZone": row.availability_zone,
            "Tags": json.dumps(row.tags),
        }


__all__ = [
    "FocusRow",
    "FocusExporter",
]
