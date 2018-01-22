from flask import Flask, request, Response
import threading
import time


class VideoServer(threading.Thread):
    def __init__(self, shared_video_filename, shared_frame, MIN_FPS=30):
        threading.Thread.__init__(self, daemon=True)

        self._app = Flask(__name__)
        self._app.route("/video_feed_<number>")(lambda number:
                                                self.video_feed(number))
        self._shared_video_filename = shared_video_filename
        self._shared_frame = shared_frame
        self.MIN_FPS = MIN_FPS

    def video_feed(self, number):
        filename = "demo_vids/{}.mov".format(number)
        self._shared_video_filename.set(filename)
        return Response(self.video_stream(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    def video_stream(self):
        start = time.time()
        while True:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + self._shared_frame.get() + b"\r\n\r\n")
            end = time.time()
            dt = end - start
            # Smoothes video stream frame rate
            if dt < 1/self.MIN_FPS:
                time.sleep(1/(self.MIN_FPS + 10) - dt)
            start = time.time()

    def run(self):
        self._app.run()

    def stop(self):
        stop_server = request.environ.get("werkzeug.server.shutdown")
        if stop_server is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        stop_server()
