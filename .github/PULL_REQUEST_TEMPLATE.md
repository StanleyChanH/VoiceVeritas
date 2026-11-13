## Summary

<!-- What does this PR change, and why? Link related issues with "Closes #123". -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation only

## Scoring-behaviour impact

<!-- If this PR changes model scoring / thresholds / features: describe the
impact on FPR@TPR95 and on under-represented groups (non-native speakers,
accents, atypical speech). Delete this section otherwise. -->

## Checklist

- [ ] Tests added/updated and passing (`uv run pytest`)
- [ ] Lint clean (`uv run ruff check src tests scripts && uv run ruff format --check src tests scripts`)
- [ ] Types clean (`uv run mypy`)
- [ ] Docs updated (README / docs/) if user-facing
- [ ] No audio files, datasets, or model weights committed
