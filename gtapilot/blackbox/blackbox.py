import io
import json
import tarfile
import time

import cv2

from gtapilot.ipc.vision_ipc import VisionIpcCpuSubscriber

SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = "blackbox-recordings"
OUTPUT_PREFIX = f"capture_{SESSION_TIMESTAMP}"
TAR_FILEPATH = f"{OUTPUT_DIR}/{OUTPUT_PREFIX}_frames.tar"
METADATA_FILEPATH = f"{OUTPUT_DIR}/{OUTPUT_PREFIX}_metadata.json"


def main():
    visionIPCSubscriber = VisionIpcCpuSubscriber()

    all_frame_metadata = []  # List to store metadata for all captured frames
    captured_frame_counter = 0

    try:
        with tarfile.open(TAR_FILEPATH, "w") as tar_out_file:
            while True:
                frame = visionIPCSubscriber.receive_frame(blocking=True)

                if frame is not None:
                    captured_frame_counter += 1

                    received_time = time.time()
                    frame_height, frame_width = frame.shape[:2]

                    # Convert frame to BGR for BMP encoding (cv2.imencode expects BGR)
                    # Assuming original_frame is RGB as per previous cvtColor logic
                    frame_bgr_for_bmp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    is_success, buffer = cv2.imencode(".bmp", frame_bgr_for_bmp)
                    if is_success:
                        bmp_bytes = buffer.tobytes()

                        # Create TarInfo for the frame
                        frame_filename_in_tar = (
                            f"frame_{captured_frame_counter:06d}.bmp"
                        )
                        tar_info = tarfile.TarInfo(name=frame_filename_in_tar)
                        tar_info.size = len(bmp_bytes)
                        tar_info.mtime = int(
                            received_time
                        )  # Set modification time in tar

                        tar_out_file.addfile(tar_info, io.BytesIO(bmp_bytes))

                        all_frame_metadata.append(
                            {
                                "timestamp": received_time,
                                "frame_archive_name": frame_filename_in_tar,
                                "resolution_height": frame_height,
                                "resolution_width": frame_width,
                            }
                        )

    finally:
        visionIPCSubscriber.close()
        if all_frame_metadata:
            # Save metadata to JSON file
            with open(METADATA_FILEPATH, "w") as metadata_file:
                json.dump(all_frame_metadata, metadata_file, indent=4)
