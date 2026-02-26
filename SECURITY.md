# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Report security vulnerabilities to **security@muveraai.com** with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested mitigations

Do **not** open a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and provide a timeline for resolution within 7 days.

## Security Design

- All database queries use parameterised statements (SQLAlchemy ORM)
- Row-level security enforced via `app.current_tenant` PostgreSQL setting
- API keys (carbon intensity provider) stored as environment variables, never in code
- Service runs as a non-root user in Docker
- No sensitive data (API keys, PII) is logged — only UUIDs and metrics
- Carbon records are append-only (no update/delete endpoints) to preserve audit trail
