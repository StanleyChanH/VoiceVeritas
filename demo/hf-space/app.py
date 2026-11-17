"""VoiceVeritas Hugging Face Space — try read-speech detection in the browser.

Deploy: push this folder to a Gradio Space (see docs/release.md). The app
installs the voiceveritas package from PyPI and fetches the same pretrained
artifact ``vv download-model`` uses (~140 KB) on cold start. Example clips
are synthesized on first boot from ``voiceveritas.utils.synth`` — the Space
repo stays text-only (HF git rejects binaries; large files need Xet).

Ethics: uploads are scored in memory and never stored or logged; the score is
decision *support* for a human reviewer, never an automatic verdict.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from voiceveritas.audio import load_audio
from voiceveritas.model_hub import download_model
from voiceveritas.models import load_model
from voiceveritas.utils.synth import SR, synthesize_speechlike

MAX_SECONDS = 120  # interview answers longer than this are scored on the first 2 min

MODEL = load_model(str(download_model()))

# same seeds as scripts/make_examples.py, so the demo clips match the repo's examples/
_EXAMPLES: dict[str, tuple[str, int]] = {
    "examples/example_read.wav": ("read", 20251116),
    "examples/example_spontaneous.wav": ("spontaneous", 20251117),
}


def _ensure_examples() -> list[str]:
    """Synthesize the demo clips on first boot (keeps the repo free of binaries)."""
    import soundfile as sf

    paths: list[str] = []
    for rel, (style, seed) in _EXAMPLES.items():
        path = Path(rel)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            clip = synthesize_speechlike(np.random.default_rng(seed), 12.0, sr=SR, style=style)
            sf.write(str(path), clip, SR, subtype="PCM_16")
        paths.append(str(path))
    return paths


_EXAMPLE_PATHS = _ensure_examples()

ETHICS_MD = """
**⚠️ Read before judging a candidate with this.**
This score is decision **support** for a human reviewer — never an automatic
verdict. The model was benchmarked on *cross-domain* data (audiobook read
speech vs meeting spontaneous speech) and **both tracks fail below ~10 dB
SNR**; treat any single number as weak evidence, recalibrate on your own
traffic, and see the [ethics charter](https://github.com/StanleyChanH/VoiceVeritas/blob/main/docs/ethics.md)
and [robustness report](https://github.com/StanleyChanH/VoiceVeritas/blob/main/docs/robustness.md).
Uploads are processed in memory and **not stored**.
"""


def predict(audio_path: str | None) -> tuple[dict, list[list], str]:
    if not audio_path:
        raise gr.Error("Upload or record an answer first (a 10–60 s clip is plenty).")
    loaded = load_audio(audio_path)
    samples, sr = loaded.samples, loaded.sample_rate
    if loaded.duration > MAX_SECONDS:  # score the first chunk, note the cut
        samples = samples[: int(MAX_SECONDS * sr)]
    result = MODEL.predict(samples, sr)

    label = "read (scripted)" if result.read_probability >= 0.5 else "spontaneous"
    conf = max(result.read_probability, 1 - result.read_probability)
    label_out = {label: float(conf)}

    contributors = [[f, round(float(c), 3)] for f, c in (result.top_contributors or [])]

    diag = [f"model: {result.model_name} · duration: {result.duration_s:.1f}s"]
    if loaded.duration > MAX_SECONDS:
        diag.append(f"input trimmed from {loaded.duration:.1f}s to {MAX_SECONDS}s")
    snr = result.estimated_snr_db
    if snr == snr:  # not NaN
        diag.append(f"estimated SNR: {snr:.1f} dB")
        if snr < 15.0:
            diag.append("⚠️ SNR < 15 dB — score unreliable, route to human review")
    for seg in result.segments:
        diag.append(f"segment [{seg.start_s:6.2f} – {seg.end_s:6.2f}s] score={seg.score:.3f}")
    return label_out, contributors, "\n".join(diag)


with gr.Blocks(title="VoiceVeritas demo") as demo:
    gr.Markdown(
        "# 🎙️ VoiceVeritas\n"
        "Was this interview answer **read from a script** or **genuinely spoken**? "
        "Upload a short answer (English speech) and get P(read) plus the prosodic "
        "cues that drove it. [GitHub](https://github.com/StanleyChanH/VoiceVeritas)"
    )
    with gr.Row():
        audio_in = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Interview answer (English, ≤2 min)",
        )
    btn = gr.Button("Score", variant="primary")
    with gr.Row():
        label_out = gr.Label(label="Verdict @ 0.5 (reference point, not policy)")
        with gr.Column():
            contrib_out = gr.Dataframe(
                headers=["prosodic cue", "contribution"],
                datatype=["str", "number"],
                label="Top drivers (interpretable Track A)",
                interactive=False,
            )
    diag_out = gr.Textbox(label="Diagnostics", lines=6, interactive=False)
    gr.Markdown(ETHICS_MD)
    btn.click(predict, inputs=audio_in, outputs=[label_out, contrib_out, diag_out])

    gr.Examples(
        examples=_EXAMPLE_PATHS,
        inputs=audio_in,
        outputs=[label_out, contrib_out, diag_out],
        fn=predict,
        cache_examples=False,  # no precompute at boot — examples score on click
        label="Synthetic caricatures (not real speech) — just to see it move",
    )


if __name__ == "__main__":
    # ssr_mode=False: Gradio's experimental Node.js SSR server crashes on
    # ZeroGPU workers and takes the app down with it (seen in run logs)
    demo.launch(ssr_mode=False)
