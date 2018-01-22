import ray
import skvideo.io


@ray.remote
class Decoder:
    def __init__(self, filename):
        self.set_video(filename)

    def next(self):
        try:
            return next(self.decoder)
        except StopIteration:
            self.decoder = skvideo.io.vreader(self.filename)
            return next(self.decoder)

    def set_video(self, filename):
        self.filename = filename
        self.decoder = skvideo.io.vreader(filename)
