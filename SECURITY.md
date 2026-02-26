# Security Policy

## Reporting a Vulnerability

Report security vulnerabilities to security@aumos.ai. Do not open public GitHub issues for security vulnerabilities.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if available)

We will acknowledge receipt within 48 hours and aim to resolve critical issues within 7 days.

## Security Considerations

### Cost Data Privacy
- Cost records contain tenant financial data — RLS enforces strict tenant isolation
- No PII is stored in cost records — only tenant_id and model_id identifiers
- Cross-tenant data access requires `get_db_session_no_tenant` with documented justification

### API Security
- All endpoints require valid JWT Bearer token
- Tenant context is extracted from the token — never from user-supplied headers alone
- Rate limiting is enforced at the API gateway layer (aumos-auth-gateway)

### Configuration
- Never commit `.env` files containing real credentials
- GPU pricing and token costs are configurable — validate inputs to prevent DoS via cost calculation loops
- Budget threshold values must be between 0.5 and 1.0 (validated by Pydantic)

### External Integrations
- OpenCost and KubeCost API responses are logged only at DEBUG level
- Raw provider metadata is stored in JSONB for audit — do not log it at INFO+ level
- HTTP timeouts are enforced on all external API calls
