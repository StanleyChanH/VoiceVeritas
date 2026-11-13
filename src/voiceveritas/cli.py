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
    p_train = sub.add_parser("train", help="train a Track-A model (prosody+LightGBM)")
    _add_common(p_train)
    p_train.add_argument("--config", required=True, help="path to a YAML TrainConfig")
    p_train.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="override/add manifests (repeatable)",
    )

    # ------------------------------------------------------------------
    p_deep = sub.add_parser("train-deep", help="train Track-B (frozen wav2vec2 + MLP)")
    _add_common(p_deep)
    p_deep.add_argument("--config", required=True, help="path to a YAML DeepTrainConfig")
    p_deep.add_argument("--manifest", action="append", default=None, help="override manifests")

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

    # ------------------------------------------------------------------
    p_prep = sub.add_parser("prepare", help="download/prepare public benchmark datasets")
    _add_common(p_prep)
    p_prep.add_argument("dataset", choices=["librispeech", "ami"], help="dataset to prepare")
    p_prep.add_argument(
        "--raw-dir", default="", help="where archives/raw audio live (default: data/raw/<dataset>)"
    )
    p_prep.add_argument(
        "--out-dir", default="", help="where processed wavs go (default: data/processed/<dataset>)"
    )
    p_prep.add_argument(
        "--manifest", default="", help="output manifest (default: data/manifests/<dataset>.jsonl)"
    )
    p_prep.add_argument("--download", action="store_true", help="download the raw data first")
    p_prep.add_argument("--max-speakers", type=int, default=20, help="librispeech: speaker cap")
    p_prep.add_argument("--max-units", type=int, default=20, help="librispeech: units per speaker")
    p_prep.add_argument("--max-chunks", type=int, default=25, help="ami: chunks per session")
    p_prep.add_argument("--seed", type=int, default=42)

    # ------------------------------------------------------------------
    p_serve = sub.add_parser("serve", help="serve the REST API (requires [api] extra)")
    _add_common(p_serve)
    p_serve.add_argument("model", help="path to model.joblib (or run directory)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--max-upload-mb", type=int, default=50)

    # ------------------------------------------------------------------
    p_ens = sub.add_parser("ensemble", help="bundle model artifacts into a median ensemble")
    _add_common(p_ens)
    p_ens.add_argument("models", nargs="+", help="two or more model.joblib paths")
    p_ens.add_argument("-o", "--out", required=True, help="output ensemble artifact path")

    # ------------------------------------------------------------------
    p_rob = sub.add_parser("robustness", help="stress a trained model under acoustic perturbations")
    _add_common(p_rob)
    p_rob.add_argument("model", help="path to model.joblib (or run directory)")
    p_rob.add_argument("--manifest", required=True, help="manifest JSONL to evaluate on")
    p_rob.add_argument("--split", default="test", help="manifest split to use (default: test)")
    p_rob.add_argument("--seed", type=int, default=42)
    p_rob.add_argument("--out", default="", help="write rows as JSON to this path")

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


def _cmd_train_deep(args: argparse.Namespace) -> int:
    from voiceveritas.config import DeepTrainConfig
    from voiceveritas.train.deep_trainer import train_deep

    config = DeepTrainConfig.from_yaml(args.config)
    if args.manifest:
        config.manifests = args.manifest
    report = train_deep(config)
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
            "estimated_snr_db": None
            if result.estimated_snr_db != result.estimated_snr_db
            else round(result.estimated_snr_db, 1),
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
    snr = result.estimated_snr_db
    if snr == snr:  # not NaN
        print(f"estimated SNR    : {snr:.1f} dB")
        if snr < 15.0:
            print("WARNING          : SNR < 15 dB — score unreliable, route to human review")
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


def _cmd_prepare(args: argparse.Namespace) -> int:
    from voiceveritas.data_prep import prepare_ami, prepare_librispeech

    dataset = args.dataset
    raw_dir = args.raw_dir or f"data/raw/{dataset}"
    out_dir = args.out_dir or f"data/processed/{dataset}"
    manifest = args.manifest or f"data/manifests/{dataset}.jsonl"

    if dataset == "librispeech":
        if args.download:
            from voiceveritas.data_prep.download import download_file
            from voiceveritas.data_prep.librispeech import LIBRISPEECH_URLS, extract_tar

            tar_path = download_file(LIBRISPEECH_URLS["dev-clean"], f"{raw_dir}/dev-clean.tar.gz")
            raw_dir = str(extract_tar(tar_path, raw_dir))
        prepare_librispeech(
            raw_dir=raw_dir,
            out_dir=out_dir,
            manifest_path=manifest,
            max_speakers=args.max_speakers,
            max_units_per_speaker=args.max_units,
            seed=args.seed,
        )
    elif dataset == "ami":
        if args.download:
            from voiceveritas.data_prep.ami import AMI_SESSIONS, download_ami_sessions

            download_ami_sessions(raw_dir, AMI_SESSIONS)
        prepare_ami(
            raw_dir=raw_dir,
            out_dir=out_dir,
            manifest_path=manifest,
            max_chunks_per_session=args.max_chunks,
            seed=args.seed,
        )
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(dataset)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed — run: uv sync --extra api", file=sys.stderr)
        return 2

    from voiceveritas.api import create_app

    model_path = (
        args.model.rstrip("/\\") + "/model.joblib"
        if not args.model.endswith("model.joblib")
        else args.model
    )
    app = create_app(model_path, max_upload_mb=args.max_upload_mb)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_robustness(args: argparse.Namespace) -> int:
    from voiceveritas.eval.robustness import run_robustness_eval
    from voiceveritas.models import load_model

    model_path = (
        args.model.rstrip("/\\") + "/model.joblib"
        if not args.model.endswith("model.joblib")
        else args.model
    )
    model = load_model(model_path)
    rows = run_robustness_eval(model, args.manifest, seed=args.seed, split=args.split)

    header = f"{'condition':<22s} {'n':>4s} {'AUC':>7s} {'EER':>7s} {'FPR@95':>7s} {'dAUC':>7s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row.get("status") == "skipped":
            print(f"{row['condition']:<22s}  skipped ({row.get('reason', '')})")
            continue
        delta = row["delta_auc_roc"]
        delta_s = "  base" if delta is None else f"{delta:+.3f}"
        print(
            f"{row['condition']:<22s} {row['n']:>4d} {row['auc_roc']:>7.3f} "
            f"{row['eer']:>7.3f} {row['fpr_at_tpr95']:>7.3f} {delta_s:>7s}"
        )

    if args.out:
        from pathlib import Path as P

        P(args.out).parent.mkdir(parents=True, exist_ok=True)
        P(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nrows written to {args.out}")
    return 0


def _cmd_ensemble(args: argparse.Namespace) -> int:
    from voiceveritas.models import load_model
    from voiceveritas.models.ensemble import EnsembleModel

    if len(args.models) < 2:
        print("an ensemble needs at least two models", file=sys.stderr)
        return 2
    members = [load_model(p) for p in args.models]
    ens = EnsembleModel(members)
    ens.save(args.out)
    print(f"ensemble of {len(members)} members saved to {args.out}")
    print("members:", ", ".join(m.name for m in members))
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
        "train-deep": _cmd_train_deep,
        "predict": _cmd_predict,
        "eval": _cmd_eval,
        "prepare": _cmd_prepare,
        "serve": _cmd_serve,
        "robustness": _cmd_robustness,
        "ensemble": _cmd_ensemble,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
