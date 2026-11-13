# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅ |

## Reporting a vulnerability

Please **do not open public issues for security vulnerabilities**.

Email the maintainers directly, or open a private security advisory via
[GitHub Security Advisories](https://github.com/StanleyChanH/VoiceVeritas/security/advisories/new).
We aim to respond within 7 days.

## Security considerations specific to this project

- The API service must not be exposed to the public internet without
  authentication, rate limiting, and TLS. It accepts file uploads by design —
  always deploy behind a gateway that enforces size limits.
- Uploaded audio is processed in-memory / temp files and is **not persisted**
  by the reference implementation. Verify this holds for your deployment.
- Model weights are code-adjacent artifacts: only load checkpoints from
  sources you trust (unpickling a malicious `.joblib`/`.pt` file executes code).
