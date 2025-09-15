import argparse
import os
from multiprocessing import freeze_support

from gtapilot.coordinator.coordinator import main as coordinator_main


def parse_args():
    parser = argparse.ArgumentParser(description="GTA Pilot coordinator entrypoint")
    parser.add_argument(
        "--video-override",
        dest="video_override",
        metavar="PATH",
        help="Path to a video file (e.g. .mp4). If provided, uses video frames instead of live display capture.",
    )
    parser.add_argument(
        "--display-id",
        dest="display_id",
        type=int,
        help="Index of the display to capture (0-based). Defaults to primary display if omitted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    freeze_support()

    args = parse_args()
    video_override = args.video_override

    if video_override:
        # Basic validation (do not attempt to open here to keep startup fast)
        if not os.path.isfile(video_override):
            raise FileNotFoundError(
                f"--video-override file not found: {video_override}"
            )

    coordinator_main(video_override=video_override, display_id=args.display_id)
