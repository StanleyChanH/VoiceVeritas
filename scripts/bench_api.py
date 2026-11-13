"""Latency/concurrency benchmark for the VoiceVeritas API.

Usage:
    uv run python scripts/bench_api.py --url http://127.0.0.1:8000 --audio demo.wav

Measures sequential latency (p50/p95) and, optionally, concurrent throughput.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from urllib import request as urlrequest

import numpy as np


def make_synthetic_wav(path: str, duration_s: float = 10.0) -> None:
    import soundfile as sf

    sr = 16000
    t = np.arange(int(duration_s * sr)) / sr
    sig = 0.3 * np.sin(2 * np.pi * 150 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    sf.write(path, sig.astype(np.float32), sr)


def post_audio(url: str, wav_path: str, timeout: float = 120.0) -> tuple[float, float]:
    boundary = "vvbenchboundary"
    with open(wav_path, "rb") as fh:
        payload = fh.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="bench.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()

    req = urlrequest.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.perf_counter()
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    wall_ms = (time.perf_counter() - started) * 1000
    return wall_ms, result["processing_ms"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", default="")
    parser.add_argument("--n", type=int, default=20, help="sequential requests")
    parser.add_argument("--duration", type=float, default=10.0, help="synthetic audio length")
    args = parser.parse_args()

    wav = args.audio
    if not wav:
        wav = "bench_input.wav"
        make_synthetic_wav(wav, args.duration)
        print(f"generated {args.duration:.0f}s synthetic wav: {wav}")

    wall_times: list[float] = []
    server_times: list[float] = []
    for i in range(args.n):
        wall, server = post_audio(f"{args.url.rstrip('/')}/v1/predict", wav)
        wall_times.append(wall)
        server_times.append(server)

    def p(xs: list[float], q: float) -> float:
        return float(np.percentile(xs, q))

    print(f"requests          : {args.n}")
    print(f"audio duration    : {args.duration:.1f}s")
    print(f"end-to-end  p50   : {p(wall_times, 50):8.1f} ms")
    print(f"end-to-end  p95   : {p(wall_times, 95):8.1f} ms")
    print(f"inference   p50   : {p(server_times, 50):8.1f} ms")
    print(f"inference   p95   : {p(server_times, 95):8.1f} ms")
    print(f"mean e2e          : {statistics.mean(wall_times):8.1f} ms")


if __name__ == "__main__":
    main()
