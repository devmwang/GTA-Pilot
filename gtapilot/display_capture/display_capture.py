import numpy as np
import dxcam


class DisplayCapture:
    def __init__(self, display=0):
        self.camera = dxcam.create(output_idx=display)
        self.camera.start()

    def getScreenshot(self):
        frame = self.camera.get_latest_frame()

        return frame

    def __del__(self):
        self.camera.stop()
