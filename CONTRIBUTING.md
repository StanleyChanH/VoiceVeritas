# Contributing to VoiceVeritas

Thanks for your interest in improving VoiceVeritas! 🎉

## First: read the ethics charter

This project sits in a sensitive domain (interview assessment / anti-cheating).
Before contributing features that affect scoring behaviour, please read
[docs/ethics.md](docs/ethics.md). In particular:

- The tool is designed to **assist human review**, never to automate rejection.
- Changes that trade false-positive rate for raw accuracy need explicit
  justification in the PR description.

## Development setup

We use [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/StanleyChanH/VoiceVeritas
cd VoiceVeritas
uv sync --extra dev --extra api --extra deep
uv run pre-commit install
```

## Workflow

1. Fork & create a feature branch (`feat/my-feature`, `fix/my-bug`).
2. Make your change with tests.
3. Ensure quality gates pass locally:

   ```bash
   uv run ruff check src tests scripts
   uv run ruff format src tests scripts
   uv run mypy
   uv run pytest
   ```

4. Open a Pull Request describing **what** and **why** (link any issue).

## Commit style

We follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `perf:`.

## Testing guidelines

- Unit tests must not require network access, GPUs, or real datasets —
  generate synthetic audio fixtures (see `tests/conftest.py`).
- Tests that download data or take >10s must be marked
  `@pytest.mark.network` / `@pytest.mark.slow`.
- Bug fixes should include a regression test.

## Adding datasets

New dataset support goes in `src/voiceveritas/manifest/` + a `scripts/prepare_*.py`.
Every dataset must document: source URL, license, speech style (read/spontaneous),
and citation. We cannot accept datasets whose license forbids redistribution of
derived artifacts without clear documentation.

## Reporting bugs / security

See [SECURITY.md](SECURITY.md) for security issues; regular bugs via
[GitHub Issues](https://github.com/StanleyChanH/VoiceVeritas/issues).
