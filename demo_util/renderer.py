import cv2
import numpy as np
import threading


class Renderer(threading.Thread):
    def __init__(self, shared_in_arrays, shared_out_frame,
                 out_shape=(720, 1280, 3), jpeg_quality=30):
        assert len(shared_in_arrays) == 4, \
            "Can only render 4 input arrays for now"
        assert out_shape[0] % 2 == 0 and out_shape[1] % 2 == 0, \
            "Output shape must be even"

        threading.Thread.__init__(self, daemon=True)

        self._shared_in_arrays = shared_in_arrays
        self._shared_out_frame = shared_out_frame
        self._out_shape = out_shape
        self._raw_frame = np.zeros(out_shape, dtype=np.uint8)
        self._jpeg_quality = jpeg_quality

    def refresh_out_frame(self):
        half_x, half_y = self._out_shape[0] // 2, self._out_shape[1] // 2
        for i, shared_in in enumerate(self._shared_in_arrays):
            partial_frame = shared_in.get()

            # Resize if necessary
            if partial_frame.shape != (half_x, half_y, 3):
                partial_frame = cv2.resize(partial_frame, (half_y, half_x))

            x_slice = slice(half_x, 2 * half_x) if i // 2 else slice(0, half_x)
            y_slice = slice(half_y, 2 * half_y) if i % 2 else slice(0, half_y)
            self._raw_frame[x_slice, y_slice] = partial_frame

        ret, jpg = cv2.imencode(".jpg", self._raw_frame,
                                [cv2.IMWRITE_JPEG_QUALITY,
                                 self._jpeg_quality])
        self._shared_out_frame.set(jpg.tobytes())

    def run(self):
        while True:
            self.refresh_out_frame()
