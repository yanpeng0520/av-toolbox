"""Command-line entry point for av-toolbox."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from av_toolbox import get_tool, list_tools, run_tool
from av_toolbox.demo_media import generate_synthetic_hiphop
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
    denseav_parser = av_subcommands.add_parser(
        "denseav",
        help="Run av.denseav.",
    )
    _add_run_arguments(denseav_parser)
    sync_parser = av_subcommands.add_parser(
        "sync-correspondence",
        help="Run av.sync_correspondence.",
    )
    _add_run_arguments(sync_parser)

    audio_parser = subcommands.add_parser("audio", help="Run audio tools.")
    audio_subcommands = audio_parser.add_subparsers(dest="audio_command", required=True)
    beat_parser = audio_subcommands.add_parser(
        "beat-detection",
        help="Run audio.beat_detection.",
    )
    _add_run_arguments(beat_parser)
    event_parser = audio_subcommands.add_parser(
        "event-detection",
        help="Run audio.event_detection.",
    )
    _add_run_arguments(event_parser)
    phase_parser = audio_subcommands.add_parser(
        "music-phase",
        help="Run audio.music_phase.",
    )
    _add_run_arguments(phase_parser)

    video_parser = subcommands.add_parser("video", help="Run video tools.")
    video_subcommands = video_parser.add_subparsers(dest="video_command", required=True)
    blur_parser = video_subcommands.add_parser(
        "blur-exposure",
        help="Run video.blur_exposure.",
    )
    _add_run_arguments(blur_parser)
    motion_parser = video_subcommands.add_parser(
        "motion",
        help="Run video.motion.",
    )
    _add_run_arguments(motion_parser)
    shot_parser = video_subcommands.add_parser(
        "shot-boundary",
        help="Run video.shot_boundary.",
    )
    _add_run_arguments(shot_parser)

    demo_parser = subcommands.add_parser("generate-demo-media", help="Generate synthetic demo WAV/MP4 media.")
    demo_parser.add_argument("--output-dir", default="data_segments")
    demo_parser.add_argument("--duration", type=float, default=60.0)
    demo_parser.add_argument("--sample-rate", type=int, default=44100)
    demo_parser.add_argument("--stem", default="synthetic_hiphop_60s")

    serve_parser = subcommands.add_parser("serve", help="Start the local web UI.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8501)
    serve_parser.add_argument("--output-root", default="outputs/web_runs")

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_path")
    parser.add_argument("--output", "--output-dir", dest="output_dir", default="outputs")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=None)
    parser.add_argument("--overlay-fps", type=float, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint", default=None)
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
        if args.av_command == "denseav":
            result = _run_from_args("av.denseav", args)
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        if args.av_command == "sync-correspondence":
            result = _run_from_args("av.sync_correspondence", args)
            print(json.dumps(result.to_dict(), indent=2))
            return 0

    if args.command == "audio":
        if args.audio_command == "beat-detection":
            result = _run_from_args("audio.beat_detection", args)
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        if args.audio_command == "event-detection":
            result = _run_from_args("audio.event_detection", args)
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        if args.audio_command == "music-phase":
            result = _run_from_args("audio.music_phase", args)
            print(json.dumps(result.to_dict(), indent=2))
            return 0

    if args.command == "video":
        if args.video_command == "blur-exposure":
            result = _run_from_args("video.blur_exposure", args)
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        if args.video_command == "motion":
            result = _run_from_args("video.motion", args)
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        if args.video_command == "shot-boundary":
            result = _run_from_args("video.shot_boundary", args)
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
        )

    return 0


def _run_from_args(tool_name: str, args: argparse.Namespace):
    return run_tool(
        tool_name,
        input_path=args.input_path,
        output_dir=args.output_dir,
        sample_fps=args.sample_fps,
        max_seconds=args.max_seconds,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        window_sec=args.window_sec,
        overlay_fps=args.overlay_fps,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        offline=args.offline,
        include_sim_matrix=args.include_sim_matrix,
        load_size=args.load_size,
        plot_size=args.plot_size,
        device=args.device,
        batch_size=args.batch_size,
        fp16=args.fp16,
        cache_dir=args.cache_dir,
        workspace_dir=args.workspace_dir,
        keep_workspace=args.keep_workspace,
        export_json=not args.no_json,
        export_csv=not args.no_csv,
        export_report=not args.no_report,
        export_overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
