import ray
import time


@ray.remote
def timer(seconds):
    time.sleep(seconds)
