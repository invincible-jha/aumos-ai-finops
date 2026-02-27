"""InvoiceGenerator adapter for AI FinOps billing reconciliation and invoice creation.

Compiles per-tenant invoices with detailed line item breakdowns, applies
tax calculations, generates invoice numbering, reconciles against provider
bills, and produces structured invoice payloads suitable for PDF rendering
or accounts payable system ingestion.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Default payment terms
DEFAULT_PAYMENT_TERMS_DAYS: int = 30

# Tax rate registry (example jurisdiction rates — overridable per tenant)
DEFAULT_TAX_RATES: dict[str, float] = {
    "US": 0.0,       # Federal — no federal sales tax; state-level varies
    "UK": 0.20,      # VAT 20%
    "EU": 0.21,      # EU average VAT
    "AU": 0.10,      # GST 10%
    "CA": 0.13,      # HST 13% (Ontario)
    "DEFAULT": 0.0,
}

# Invoice line item categories
LINE_ITEM_COMPUTE = "compute"
LINE_ITEM_STORAGE = "storage"
LINE_ITEM_INFERENCE = "inference_tokens"
LINE_ITEM_NETWORK = "network_egress"
LINE_ITEM_SUPPORT = "support"
LINE_ITEM_PLATFORM = "platform_fee"
LINE_ITEM_DISCOUNT = "discount"


class InvoiceGenerator:
    """AI FinOps billing reconciliation and invoice generation engine.

    Compiles structured invoices from cost records and token usage data,
    applies configurable tax rates, generates sequential invoice numbers,
    and reconciles computed invoices against provider bills to surface
    discrepancies. All invoice data is tenant-scoped.

    The generator is a pure computation adapter — it receives cost data
    as structured dicts from the service layer and produces invoice objects
    without touching the database directly.
    """

    def __init__(
        self,
        payment_terms_days: int = DEFAULT_PAYMENT_TERMS_DAYS,
        invoice_prefix: str = "AUMOS",
        tax_rates: dict[str, float] | None = None,
        currency: str = "USD",
    ) -> None:
        """Initialise the InvoiceGenerator.

        Args:
            payment_terms_days: Days from invoice date to payment due date.
            invoice_prefix: Prefix for generated invoice numbers.
            tax_rates: Override tax rate registry (jurisdiction -> rate).
            currency: Currency code for all invoice amounts.
        """
        self._payment_terms_days = payment_terms_days
        self._invoice_prefix = invoice_prefix
        self._tax_rates = tax_rates or DEFAULT_TAX_RATES
        self._currency = currency
        self._invoice_sequence: dict[str, int] = {}

    async def compile_tenant_invoice(
        self,
        tenant_id: str,
        tenant_name: str,
        period_start: datetime,
        period_end: datetime,
        cost_records: list[dict[str, Any]],
        token_usage_records: list[dict[str, Any]],
        tax_jurisdiction: str = "DEFAULT",
        discount_percent: float = 0.0,
        support_tier: str | None = None,
        platform_fee_usd: float = 0.0,
        billing_address: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compile a complete invoice for a tenant's billing period.

        Args:
            tenant_id: Tenant identifier.
            tenant_name: Tenant organization name.
            period_start: Billing period start (UTC).
            period_end: Billing period end (UTC).
            cost_records: List of cost record dicts with resource_type, cost_usd.
            token_usage_records: List of token usage dicts with model_name,
                total_tokens, total_cost_usd.
            tax_jurisdiction: Jurisdiction code for tax rate lookup.
            discount_percent: Volume or contractual discount percent (0–100).
            support_tier: Optional support tier name for support line item.
            platform_fee_usd: Fixed platform fee to include.
            billing_address: Optional billing address dict for the invoice header.

        Returns:
            Complete invoice dict with header, line_items, subtotal, tax,
            discount, total, and payment_terms sections.
        """
        logger.info(
            "Compiling tenant invoice",
            tenant_id=tenant_id,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            cost_record_count=len(cost_records),
            token_usage_count=len(token_usage_records),
        )

        line_items = self._build_line_items(
            cost_records=cost_records,
            token_usage_records=token_usage_records,
            support_tier=support_tier,
            platform_fee_usd=platform_fee_usd,
        )

        subtotal = sum(item["amount_usd"] for item in line_items if item["category"] != LINE_ITEM_DISCOUNT)

        discount_amount = round(subtotal * (discount_percent / 100.0), 4) if discount_percent > 0 else 0.0
        if discount_amount > 0:
            line_items.append({
                "line_item_id": str(uuid.uuid4()),
                "category": LINE_ITEM_DISCOUNT,
                "description": f"Volume/contractual discount ({discount_percent:.1f}%)",
                "quantity": 1.0,
                "unit": "lump sum",
                "unit_price_usd": -discount_amount,
                "amount_usd": -discount_amount,
            })

        taxable_amount = subtotal - discount_amount
        tax_rate = self._tax_rates.get(tax_jurisdiction, self._tax_rates["DEFAULT"])
        tax_amount = round(taxable_amount * tax_rate, 4)

        total = round(taxable_amount + tax_amount, 2)

        invoice_number = self._generate_invoice_number(tenant_id, period_start)
        invoice_date = datetime.now(tz=timezone.utc)

        import datetime as dt
        due_date = invoice_date + dt.timedelta(days=self._payment_terms_days)

        invoice: dict[str, Any] = {
            "invoice_id": str(uuid.uuid4()),
            "invoice_number": invoice_number,
            "invoice_date": invoice_date.date().isoformat(),
            "due_date": due_date.date().isoformat(),
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "billing_address": billing_address or {},
            "currency": self._currency,
            "billing_period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
                "period_label": f"{period_start.strftime('%B %Y')}",
            },
            "line_items": line_items,
            "financial_summary": {
                "subtotal_usd": round(subtotal, 2),
                "discount_usd": round(discount_amount, 2),
                "discount_percent": discount_percent,
                "taxable_amount_usd": round(taxable_amount, 2),
                "tax_jurisdiction": tax_jurisdiction,
                "tax_rate_percent": round(tax_rate * 100.0, 2),
                "tax_amount_usd": round(tax_amount, 2),
                "total_usd": total,
            },
            "payment_terms": {
                "days": self._payment_terms_days,
                "due_date": due_date.date().isoformat(),
                "accepted_methods": ["bank_transfer", "credit_card", "ach"],
                "late_fee_percent": 1.5,
            },
            "status": "draft",
            "generated_at": invoice_date.isoformat(),
        }

        logger.info(
            "Tenant invoice compiled",
            tenant_id=tenant_id,
            invoice_number=invoice_number,
            total_usd=total,
            line_item_count=len(line_items),
        )

        return invoice

    def _build_line_items(
        self,
        cost_records: list[dict[str, Any]],
        token_usage_records: list[dict[str, Any]],
        support_tier: str | None,
        platform_fee_usd: float,
    ) -> list[dict[str, Any]]:
        """Build structured invoice line items from cost and usage records.

        Args:
            cost_records: Cost record dicts grouped and summed by resource_type.
            token_usage_records: Token usage dicts with model-level breakdowns.
            support_tier: Optional support tier label.
            platform_fee_usd: Fixed platform fee amount.

        Returns:
            List of line item dicts with description, quantity, unit_price, amount.
        """
        line_items: list[dict[str, Any]] = []

        # Aggregate cost records by resource type
        from collections import defaultdict
        cost_by_type: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_usd": 0.0, "count": 0, "gpu_hours": 0.0}
        )

        for record in cost_records:
            resource_type = record.get("resource_type", "compute")
            cost_by_type[resource_type]["total_usd"] += float(record.get("cost_usd", 0.0))
            cost_by_type[resource_type]["count"] += 1
            if resource_type == "gpu":
                cost_by_type[resource_type]["gpu_hours"] += float(record.get("gpu_hours", 1.0))

        # GPU / compute line items
        if "gpu" in cost_by_type:
            gpu_data = cost_by_type.pop("gpu")
            gpu_hours = gpu_data.get("gpu_hours") or float(gpu_data["count"])
            unit_price = round(gpu_data["total_usd"] / gpu_hours, 4) if gpu_hours > 0 else 0.0
            line_items.append({
                "line_item_id": str(uuid.uuid4()),
                "category": LINE_ITEM_COMPUTE,
                "description": "GPU Compute (measured in GPU-hours)",
                "quantity": round(gpu_hours, 2),
                "unit": "GPU-hour",
                "unit_price_usd": unit_price,
                "amount_usd": round(gpu_data["total_usd"], 4),
            })

        for resource_type, data in cost_by_type.items():
            if data["total_usd"] == 0.0:
                continue
            category = (
                LINE_ITEM_STORAGE if "storage" in resource_type
                else LINE_ITEM_NETWORK if "network" in resource_type
                else LINE_ITEM_COMPUTE
            )
            line_items.append({
                "line_item_id": str(uuid.uuid4()),
                "category": category,
                "description": f"{resource_type.replace('_', ' ').title()} Usage",
                "quantity": float(data["count"]),
                "unit": "resource-hour",
                "unit_price_usd": round(data["total_usd"] / data["count"], 4),
                "amount_usd": round(data["total_usd"], 4),
            })

        # Token inference line items — one per model
        for usage in token_usage_records:
            model_name = usage.get("model_name", "Unknown Model")
            total_tokens = int(usage.get("total_tokens", 0))
            total_cost = float(usage.get("total_cost_usd", 0.0))
            if total_tokens == 0:
                continue
            cost_per_million = round(total_cost / (total_tokens / 1_000_000), 4) if total_tokens > 0 else 0.0
            line_items.append({
                "line_item_id": str(uuid.uuid4()),
                "category": LINE_ITEM_INFERENCE,
                "description": f"LLM Inference — {model_name}",
                "quantity": round(total_tokens / 1_000_000, 4),
                "unit": "M tokens",
                "unit_price_usd": cost_per_million,
                "amount_usd": round(total_cost, 4),
            })

        # Support tier fee
        if support_tier:
            support_fees = {
                "basic": 0.0,
                "standard": 500.0,
                "premium": 2000.0,
                "enterprise": 5000.0,
            }
            support_fee = support_fees.get(support_tier.lower(), 0.0)
            if support_fee > 0:
                line_items.append({
                    "line_item_id": str(uuid.uuid4()),
                    "category": LINE_ITEM_SUPPORT,
                    "description": f"{support_tier.title()} Support Tier",
                    "quantity": 1.0,
                    "unit": "month",
                    "unit_price_usd": support_fee,
                    "amount_usd": support_fee,
                })

        # Platform fee
        if platform_fee_usd > 0:
            line_items.append({
                "line_item_id": str(uuid.uuid4()),
                "category": LINE_ITEM_PLATFORM,
                "description": "AumOS Platform Fee",
                "quantity": 1.0,
                "unit": "month",
                "unit_price_usd": platform_fee_usd,
                "amount_usd": platform_fee_usd,
            })

        return line_items

    async def generate_pdf_invoice_data(
        self,
        invoice: dict[str, Any],
        vendor_details: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate structured data for PDF invoice rendering.

        Args:
            invoice: Compiled invoice dict from compile_tenant_invoice.
            vendor_details: Issuing vendor/company details dict with:
                - company_name: str
                - address: dict
                - logo_url: str | None
                - tax_id: str | None
                - bank_details: dict

        Returns:
            PDF-ready structured data dict with all sections formatted for
            a document renderer.
        """
        logger.info(
            "Generating PDF invoice data",
            invoice_number=invoice.get("invoice_number"),
            tenant_id=invoice.get("tenant_id"),
        )

        financial = invoice.get("financial_summary", {})
        line_items = invoice.get("line_items", [])

        # Group line items by category for PDF rendering
        from collections import defaultdict
        items_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in line_items:
            items_by_category[item["category"]].append(item)

        formatted_line_items: list[dict[str, Any]] = []
        for category, items in items_by_category.items():
            if items:
                formatted_line_items.append({
                    "section_header": category.replace("_", " ").title(),
                    "items": [
                        {
                            "description": item["description"],
                            "quantity": f"{item['quantity']:,.4f} {item['unit']}",
                            "unit_price": f"${item['unit_price_usd']:,.4f}",
                            "amount": f"${item['amount_usd']:,.2f}",
                        }
                        for item in items
                    ],
                    "section_total": f"${sum(i['amount_usd'] for i in items):,.2f}",
                })

        pdf_data: dict[str, Any] = {
            "document_type": "invoice",
            "invoice_number": invoice.get("invoice_number"),
            "invoice_date": invoice.get("invoice_date"),
            "due_date": invoice.get("due_date"),
            "issuer": {
                "company_name": vendor_details.get("company_name", "AumOS Platform"),
                "address": vendor_details.get("address", {}),
                "logo_url": vendor_details.get("logo_url"),
                "tax_id": vendor_details.get("tax_id"),
            },
            "recipient": {
                "tenant_name": invoice.get("tenant_name"),
                "billing_address": invoice.get("billing_address", {}),
                "tenant_id": invoice.get("tenant_id"),
            },
            "billing_period": invoice.get("billing_period", {}),
            "line_item_sections": formatted_line_items,
            "financial_totals": {
                "subtotal": f"${financial.get('subtotal_usd', 0.0):,.2f}",
                "discount": f"-${financial.get('discount_usd', 0.0):,.2f}" if financial.get("discount_usd") else None,
                "tax": f"${financial.get('tax_amount_usd', 0.0):,.2f} ({financial.get('tax_rate_percent', 0.0):.1f}% {financial.get('tax_jurisdiction', '')})",
                "total": f"${financial.get('total_usd', 0.0):,.2f} {invoice.get('currency', 'USD')}",
            },
            "payment_instructions": {
                "due_date": invoice.get("due_date"),
                "accepted_methods": invoice.get("payment_terms", {}).get("accepted_methods", []),
                "bank_details": vendor_details.get("bank_details", {}),
                "late_fee_notice": f"A {invoice.get('payment_terms', {}).get('late_fee_percent', 1.5)}% monthly late fee applies after the due date.",
            },
            "footer": {
                "generated_by": "AumOS AI FinOps",
                "generated_at": invoice.get("generated_at"),
            },
        }

        logger.info(
            "PDF invoice data generated",
            invoice_number=invoice.get("invoice_number"),
        )

        return pdf_data

    async def reconcile_with_provider_bill(
        self,
        tenant_id: str,
        computed_invoice: dict[str, Any],
        provider_bill: dict[str, Any],
        tolerance_percent: float = 2.0,
    ) -> dict[str, Any]:
        """Reconcile a computed invoice against a provider's bill.

        Args:
            tenant_id: Tenant identifier.
            computed_invoice: Invoice dict from compile_tenant_invoice.
            provider_bill: Provider bill dict with:
                - provider_name: str
                - bill_number: str
                - total_usd: float
                - line_items: list[dict] (provider's breakdown)
            tolerance_percent: Acceptable variance percent before flagging.

        Returns:
            Dict with is_reconciled, variance_usd, variance_percent,
            discrepancies (list), and reconciliation_status fields.
        """
        logger.info(
            "Reconciling invoice with provider bill",
            tenant_id=tenant_id,
            invoice_number=computed_invoice.get("invoice_number"),
            provider=provider_bill.get("provider_name"),
        )

        computed_total = computed_invoice.get("financial_summary", {}).get("total_usd", 0.0)
        provider_total = float(provider_bill.get("total_usd", 0.0))

        variance_usd = round(computed_total - provider_total, 2)
        variance_percent = round(
            abs(variance_usd) / provider_total * 100.0, 4
        ) if provider_total > 0 else 0.0

        is_reconciled = variance_percent <= tolerance_percent
        discrepancies: list[dict[str, Any]] = []

        if not is_reconciled:
            discrepancies.append({
                "type": "total_mismatch",
                "severity": "high" if variance_percent > tolerance_percent * 2 else "medium",
                "computed_usd": computed_total,
                "provider_usd": provider_total,
                "variance_usd": variance_usd,
                "variance_percent": variance_percent,
                "description": (
                    f"Total invoice amount mismatch: computed ${computed_total:,.2f}, "
                    f"provider ${provider_total:,.2f} "
                    f"({variance_percent:.2f}% variance)."
                ),
            })

        # Line item reconciliation
        computed_items = computed_invoice.get("line_items", [])
        provider_items = provider_bill.get("line_items", [])

        computed_categories: dict[str, float] = {}
        for item in computed_items:
            cat = item.get("category", "unknown")
            computed_categories[cat] = computed_categories.get(cat, 0.0) + item.get("amount_usd", 0.0)

        provider_categories: dict[str, float] = {}
        for item in provider_items:
            cat = item.get("category", "unknown")
            provider_categories[cat] = provider_categories.get(cat, 0.0) + float(item.get("amount_usd", 0.0))

        all_categories = set(computed_categories.keys()) | set(provider_categories.keys())
        for category in all_categories:
            comp_amount = computed_categories.get(category, 0.0)
            prov_amount = provider_categories.get(category, 0.0)
            cat_variance = comp_amount - prov_amount

            if abs(cat_variance) > 0.01 and (
                prov_amount == 0 or abs(cat_variance / prov_amount) * 100.0 > tolerance_percent
            ):
                discrepancies.append({
                    "type": "line_item_mismatch",
                    "category": category,
                    "severity": "medium",
                    "computed_usd": round(comp_amount, 2),
                    "provider_usd": round(prov_amount, 2),
                    "variance_usd": round(cat_variance, 2),
                    "description": f"Category '{category}': computed ${comp_amount:,.2f} vs provider ${prov_amount:,.2f}.",
                })

        reconciliation_status = (
            "reconciled" if is_reconciled and not discrepancies
            else "minor_discrepancy" if variance_percent <= tolerance_percent
            else "major_discrepancy"
        )

        result: dict[str, Any] = {
            "reconciliation_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "invoice_number": computed_invoice.get("invoice_number"),
            "provider_name": provider_bill.get("provider_name"),
            "provider_bill_number": provider_bill.get("bill_number"),
            "computed_total_usd": computed_total,
            "provider_total_usd": provider_total,
            "variance_usd": variance_usd,
            "variance_percent": variance_percent,
            "tolerance_percent": tolerance_percent,
            "is_reconciled": is_reconciled,
            "reconciliation_status": reconciliation_status,
            "discrepancies": discrepancies,
            "discrepancy_count": len(discrepancies),
            "reconciled_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        logger.info(
            "Invoice reconciliation completed",
            tenant_id=tenant_id,
            reconciliation_status=reconciliation_status,
            variance_usd=variance_usd,
            discrepancy_count=len(discrepancies),
        )

        return result

    def _generate_invoice_number(
        self,
        tenant_id: str,
        period_start: datetime,
    ) -> str:
        """Generate a sequential invoice number for a tenant and period.

        Uses a deterministic hash of tenant_id + year/month to ensure
        idempotent invoice numbering within a billing period.

        Args:
            tenant_id: Tenant identifier.
            period_start: Billing period start for the invoice.

        Returns:
            Invoice number string (e.g., "AUMOS-2024-01-ABCD-0001").
        """
        period_key = period_start.strftime("%Y-%m")
        sequence_key = f"{tenant_id}:{period_key}"

        if sequence_key not in self._invoice_sequence:
            self._invoice_sequence[sequence_key] = 0
        self._invoice_sequence[sequence_key] += 1

        sequence_num = self._invoice_sequence[sequence_key]

        # Short tenant hash for readability
        tenant_hash = hashlib.md5(tenant_id.encode()).hexdigest()[:6].upper()

        return f"{self._invoice_prefix}-{period_key}-{tenant_hash}-{sequence_num:04d}"
