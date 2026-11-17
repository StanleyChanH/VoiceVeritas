# Release checklist / 发布清单

Every release ships four things in sync: the **git tag**, the **GitHub
release** (with the pretrained artifact attached), the **PyPI package**, and
(optionally, when the model changed) the **Hugging Face Space demo**.

## 1. One-time setup (already done? skip)

- [ ] **PyPI trusted publishing** — at <https://pypi.org/manage/account/publishing/>
  add a *pending* publisher: project `voiceveritas`, owner `StanleyChanH`,
  repository `VoiceVeritas`, workflow name `pypi.yml` (the workflow **file
  name**, not its `name:` field), environment `pypi`. Requires 2FA on the
  PyPI account.
- [ ] **GitHub release environment** — repo Settings → Environments → create
  `pypi` (referenced by `.github/workflows/pypi.yml`; no secrets needed).
- [ ] **Hugging Face Space** — create a Gradio Space, then push `demo/hf-space/`
  (see the deploy section at the bottom).

## 2. Every release

1. Bump `version` in `pyproject.toml`; make sure the CLI/API version warnings
   still make sense (`vv` warns when model package ≠ runtime package).
2. Update `DEFAULT_VERSION` in `src/voiceveritas/model_hub.py` **if** the new
   release attaches a new pretrained artifact users should get by default.
3. Commit, tag, push:
   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
4. Create the GitHub release from the tag. If the pretrained model changed,
   retrain, then attach the artifact named
   `voiceveritas-<model>-vX.Y.Z.joblib` (the name pattern **must** match
   `_MODELS` in `model_hub.py` — the asset embeds the full tag including the
   leading `v`, e.g. `voiceveritas-prosody-lgbm-v0.1.2.joblib`).
5. Publishing the release triggers `Publish to PyPI` (build → twine check →
   trusted-publishing upload). Verify on
   <https://pypi.org/project/voiceveritas/>.
6. Post-release smoke test (do this *before* announcing anything) — from a
   clean environment, no repo checkout needed:
   ```bash
   pip install --upgrade pipx && pipx run --spec voiceveritas vv download-model
   # score a freshly synthesized clip using only the installed package:
   python - <<'PY'
   import numpy as np, soundfile as sf
   from voiceveritas.utils.synth import synthesize_speechlike
   sf.write("smoke.wav", synthesize_speechlike(np.random.default_rng(1), 8.0, style="read"), 16000)
   PY
   pipx run --spec voiceveritas vv predict "$(pipx run --spec voiceveritas vv download-model)" smoke.wav
   ```

## 3. Hugging Face Space deploy

The demo lives in [`demo/hf-space/`](../demo/hf-space/) and is a normal
Gradio Space that installs `voiceveritas` from PyPI and fetches the same
GitHub release artifact on startup.

```bash
# first deploy (or updates): copy the folder into a clone of the Space repo
git clone https://huggingface.co/spaces/<user>/voiceveritas hf-space
cp demo/hf-space/* hf-space/ -r
cd hf-space && git add . && git commit -m "update demo" && git push
```

Update `requirements.txt` in the Space when a new package version should be
pinned.

## 4. Announcing (promotion)

Only after the post-release smoke test passes: update the README badges if
needed, then follow the promotion plan (Show HN / V2EX / Reddit / Twitter,
spaced out over weeks, not all at once).
