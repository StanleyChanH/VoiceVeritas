"""Command-line interface: ``vv <subcommand>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vv",
        description="VoiceVeritas: read-speech detection for interview audio.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    p_train = sub.add_parser("train", help="train a model from a config file")
    _add_common(p_train)
    p_train.add_argument("--config", required=True, help="path to a YAML TrainConfig")
    p_train.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="override/add manifests (repeatable)",
    )

    # ------------------------------------------------------------------
    p_predict = sub.add_parser("predict", help="score one audio file")
    _add_common(p_predict)
    p_predict.add_argument("model", help="path to model.joblib")
    p_predict.add_argument("audio", help="path to audio file (.wav/.mp3/.m4a)")
    p_predict.add_argument("--json", action="store_true", help="machine-readable output")

    # ------------------------------------------------------------------
    p_eval = sub.add_parser("eval", help="evaluate a trained run on a manifest")
    _add_common(p_eval)
    p_eval.add_argument("--run", required=True, help="run directory (contains model.joblib)")
    p_eval.add_argument("--manifest", required=True, help="manifest JSONL to evaluate on")

    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    from voiceveritas.config import TrainConfig
    from voiceveritas.train import train_baseline

    config = TrainConfig.from_yaml(args.config)
    if args.manifest:
        config.manifests = args.manifest
    report = train_baseline(config)
    print(json.dumps(report, indent=2, default=str))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from voiceveritas.audio import load_audio
    from voiceveritas.models import load_model

    model = load_model(args.model)
    loaded = load_audio(args.audio)
    result = model.predict(loaded.samples, loaded.sample_rate)

    if args.json:
        payload = {
            "audio": args.audio,
            "read_probability": result.read_probability,
            "aggregation": result.aggregation,
            "duration_s": result.duration_s,
            "model": result.model_name,
            "segments": [
                {"start_s": s.start_s, "end_s": s.end_s, "score": s.score} for s in result.segments
            ],
        }
        if result.top_contributors:
            payload["top_contributors"] = [
                {"feature": f, "contribution": c} for f, c in result.top_contributors
            ]
        print(json.dumps(payload, indent=2))
        return 0

    verdict = "READ" if result.read_probability >= 0.5 else "SPONTANEOUS"
    print(f"model            : {result.model_name}")
    print(f"duration         : {result.duration_s:.1f}s")
    print(f"P(read)          : {result.read_probability:.3f}")
    print(f"verdict@0.5      : {verdict}")
    for seg in result.segments:
        print(f"  [{seg.start_s:7.2f} - {seg.end_s:7.2f}s]  score={seg.score:.3f}")
    if result.top_contributors:
        print("top contributors :")
        for feat, contrib in result.top_contributors:
            print(f"  {feat:<24s} {contrib:+.3f}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from voiceveritas.audio import load_audio
    from voiceveritas.manifest import load_manifest
    from voiceveritas.metrics import compute_binary_metrics
    from voiceveritas.models import load_model

    model = load_model(
        args.run.rstrip("/\\") + "/model.joblib"
        if not args.run.endswith("model.joblib")
        else args.run
    )
    utts = load_manifest(args.manifest)
    y_true, scores = [], []
    for utt in utts:
        try:
            loaded = load_audio(utt.path)
        except Exception as exc:
            logging.warning("skipping %s: %s", utt.utt_id, exc)
            continue
        result = model.predict(loaded.samples, loaded.sample_rate)
        y_true.append(utt.label)
        scores.append(result.read_probability)
    if not y_true:
        print("no utterances evaluated", file=sys.stderr)
        return 1
    print(json.dumps(compute_binary_metrics(y_true, scores, label="eval"), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    handlers = {
        "train": _cmd_train,
        "predict": _cmd_predict,
        "eval": _cmd_eval,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
