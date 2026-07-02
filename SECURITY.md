# Security Policy

## Supported versions

`av-toolbox` is pre-1.0 and under active development. Security fixes are applied
to the latest `main` and the most recent release only.

| Version | Supported |
| --- | --- |
| latest `main` / newest release | ✅ |
| older releases | ❌ |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Use GitHub's private vulnerability reporting instead:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** and fill in the advisory form.

Please include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal proof-of-concept if possible).
- Affected version/commit and environment details.

We aim to acknowledge reports within a few days and to coordinate a fix and
disclosure timeline with you. This is a small, best-effort project, so please
allow reasonable time before any public disclosure.

## Scope and notes

- `av-toolbox` runs local media analysis and can start a local/hosted Streamlit
  UI. The public demo mode enforces upload guardrails (max upload size and
  analyzed duration); when self-hosting, keep those limits and run behind your
  own network controls.
- Model-backed tools download third-party weights (e.g. YOLO, TransNetV2,
  Whisper, PyTorchVideo) on first use. Only enable the extras you trust, and
  point `AV_TOOLBOX_CACHE_DIR` at a location you control.
- Never commit secrets. Runtime secrets belong in gitignored `.env` files, not
  in the repository.
