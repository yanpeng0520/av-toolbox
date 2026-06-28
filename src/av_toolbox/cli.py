"""Command-line entry point for av-toolbox."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from av_toolbox import get_tool, list_tools, run_tool
from av_toolbox.demo_media import generate_synthetic_hiphop
from av_toolbox.public_demo import DEFAULT_PUBLIC_MAX_SECONDS, DEFAULT_PUBLIC_MAX_UPLOAD_MB
from av_toolbox.web import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="av-toolbox")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("list-tools", help="List registered tools.")

    info_parser = subcommands.add_parser("info", help="Show one registered tool.")
    info_parser.add_argument("tool_name")

    run_parser = subcommands.add_parser("run", help="Run a registered tool.")
    run_parser.add_argument("tool_name")
    _add_run_arguments(run_parser)

    av_parser = subcommands.add_parser("av", help="Run audio-visual tools.")
    av_subcommands = av_parser.add_subparsers(dest="av_command", required=True)
    _add_tool_command(av_subcommands, "denseav", "Run av.denseav.")
    _add_tool_command(av_subcommands, "sync-correspondence", "Run av.sync_correspondence.")

    audio_parser = subcommands.add_parser("audio", help="Run audio tools.")
    audio_subcommands = audio_parser.add_subparsers(dest="audio_command", required=True)
    _add_tool_command(audio_subcommands, "beat-detection", "Run audio.beat_detection.")
    _add_tool_command(audio_subcommands, "energy", "Run audio.energy.")
    _add_tool_command(audio_subcommands, "event-detection", "Run audio.event_detection.")
    _add_tool_command(audio_subcommands, "music-phase", "Run audio.music_phase.")
    _add_tool_command(audio_subcommands, "transcription", "Run audio.transcription.")

    video_parser = subcommands.add_parser("video", help="Run video tools.")
    video_subcommands = video_parser.add_subparsers(dest="video_command", required=True)
    _add_tool_command(video_subcommands, "action-recognition", "Run video.action_recognition.")
    _add_tool_command(video_subcommands, "blur-exposure", "Run video.blur_exposure.")
    _add_tool_command(video_subcommands, "camera-shake", "Run video.camera_shake.")
    _add_tool_command(video_subcommands, "cut-detection", "Run video.cut_detection.")
    _add_tool_command(video_subcommands, "foreground-motion", "Run video.foreground_motion.")
    _add_tool_command(video_subcommands, "motion", "Run video.motion.")
    _add_tool_command(video_subcommands, "object-detection", "Run video.object_detection.")
    _add_tool_command(video_subcommands, "obstruction", "Run video.obstruction.")
    _add_tool_command(video_subcommands, "optical-flow", "Run video.optical_flow.")
    _add_tool_command(video_subcommands, "pose", "Run video.pose.")
    _add_tool_command(video_subcommands, "segmentation", "Run video.segmentation.")
    _add_tool_command(video_subcommands, "shot-boundary", "Run video.shot_boundary.")
    _add_tool_command(video_subcommands, "shot-type", "Run video.shot_type.")
    _add_tool_command(video_subcommands, "st-action", "Run video.st_action.")

    demo_parser = subcommands.add_parser("generate-demo-media", help="Generate synthetic demo WAV/MP4 media.")
    demo_parser.add_argument("--output-dir", default="data_segments")
    demo_parser.add_argument("--duration", type=float, default=60.0)
    demo_parser.add_argument("--sample-rate", type=int, default=44100)
    demo_parser.add_argument("--stem", default="synthetic_hiphop_60s")

    serve_parser = subcommands.add_parser("serve", help="Start the local web UI.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8501)
    serve_parser.add_argument("--output-root", default="outputs/web_runs")
    serve_parser.add_argument("--page-title", default=None, help="Browser and visible page title for the web UI.")
    serve_parser.add_argument("--public-demo", action="store_true", help="Start with public upload-demo guardrails.")
    serve_parser.add_argument("--public-max-seconds", type=float, default=DEFAULT_PUBLIC_MAX_SECONDS)
    serve_parser.add_argument("--public-max-upload-mb", type=int, default=DEFAULT_PUBLIC_MAX_UPLOAD_MB)
    serve_parser.add_argument("--public-enable-denseav", action="store_true", help="Include DenseAV in public demo workflows.")

    return parser


def _add_tool_command(subcommands: argparse._SubParsersAction, name: str, help_text: str) -> None:
    command = subcommands.add_parser(name, help=help_text)
    _add_run_arguments(command)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_path")
    parser.add_argument("--output", "--output-dir", dest="output_dir", default="outputs")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--frame-length", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--overlay-fps", type=float, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--labels-path", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--mask-mode", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--step-seconds", type=float, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--include-sim-matrix", action="store_true")
    parser.add_argument("--load-size", type=int, default=None)
    parser.add_argument("--plot-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--no-vad-filter", dest="vad_filter", action="store_false", default=None)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-tools":
        print(json.dumps(list_tools(), indent=2))
        return 0

    if args.command == "info":
        tool = get_tool(args.tool_name)
        print(json.dumps({
            "name": tool.name,
            "category": getattr(tool, "category", ""),
            "description": getattr(tool, "description", ""),
        }, indent=2))
        return 0

    if args.command == "run":
        result = _run_from_args(args.tool_name, args)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "av":
        tool_name = {
            "denseav": "av.denseav",
            "sync-correspondence": "av.sync_correspondence",
        }[args.av_command]
        result = _run_from_args(tool_name, args)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "audio":
        tool_name = {
            "beat-detection": "audio.beat_detection",
            "energy": "audio.energy",
            "event-detection": "audio.event_detection",
            "music-phase": "audio.music_phase",
            "transcription": "audio.transcription",
        }[args.audio_command]
        result = _run_from_args(tool_name, args)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "video":
        tool_name = {
            "action-recognition": "video.action_recognition",
            "blur-exposure": "video.blur_exposure",
            "camera-shake": "video.camera_shake",
            "cut-detection": "video.cut_detection",
            "foreground-motion": "video.foreground_motion",
            "motion": "video.motion",
            "object-detection": "video.object_detection",
            "obstruction": "video.obstruction",
            "optical-flow": "video.optical_flow",
            "pose": "video.pose",
            "segmentation": "video.segmentation",
            "shot-boundary": "video.shot_boundary",
            "shot-type": "video.shot_type",
            "st-action": "video.st_action",
        }[args.video_command]
        result = _run_from_args(tool_name, args)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "generate-demo-media":
        payload = generate_synthetic_hiphop(
            output_dir=args.output_dir,
            duration=args.duration,
            sample_rate=args.sample_rate,
            stem=args.stem,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "serve":
        return serve(
            host=args.host,
            port=args.port,
            output_root=args.output_root,
            public_demo=args.public_demo,
            public_max_seconds=args.public_max_seconds,
            public_max_upload_mb=args.public_max_upload_mb,
            public_enable_denseav=args.public_enable_denseav,
            page_title=args.page_title,
        )

    return 0


def _run_from_args(tool_name: str, args: argparse.Namespace):
    kwargs = {
        "input_path": args.input_path,
        "output_dir": args.output_dir,
        "sample_fps": args.sample_fps,
        "max_seconds": args.max_seconds,
        "offline": args.offline,
        "include_sim_matrix": args.include_sim_matrix,
        "device": args.device,
        "batch_size": args.batch_size,
        "fp16": args.fp16,
        "cache_dir": args.cache_dir,
        "workspace_dir": args.workspace_dir,
        "keep_workspace": args.keep_workspace,
        "export_json": not args.no_json,
        "export_csv": not args.no_csv,
        "export_report": not args.no_report,
        "export_overlay": not args.no_overlay,
    }
    optional_names = [
        "sample_rate",
        "hop_length",
        "frame_length",
        "window_sec",
        "overlay_fps",
        "model_name",
        "checkpoint",
        "config_path",
        "labels_path",
        "language",
        "compute_type",
        "backend",
        "mask_mode",
        "threshold",
        "confidence",
        "image_size",
        "top_k",
        "beam_size",
        "window_seconds",
        "step_seconds",
        "load_size",
        "plot_size",
        "vad_filter",
    ]
    for name in optional_names:
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    return run_tool(tool_name, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
