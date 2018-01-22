import numpy as np
import os
import ray
import sys

from demo_util.renderer import Renderer
from demo_util.sync import SharedObject
from demo_util.timer import timer
from demo_util.video_server import VideoServer

# Environment setup
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + \
        ":./BDD_Driving_Model"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":./drn/"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") \
        + ":./tf_models/research/"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") \
        + ":./tf_models/research/object_detection/"

sys.path.append("BDD_Driving_Model")
sys.path.append("drn")
sys.path.append("tf_models/research")
sys.path.append("tf_models/research/object_detection")

# Imports requiring added paths
from actors.decoder import Decoder
from actors.driving_model import DrivingModel
from actors.object_detector import ObjectDetector
from actors.segmentor import Segmentor
import segment


ray.init()

# Initialize LSTM actor
lstm = DrivingModel.remote("discrete_cnn_lstm",
                           "data/discrete_cnn_lstm_py3/model.ckpt", 1)

# Initialize segmentation actor
seg_actor = Segmentor.remote("drn_d_22", 19, "data/drn_d_22_bdd_v1.pth",
                             segment.CITYSCAPE_PALLETE)

# Initialize object detection actor
OD_MODEL_NAME = "data/ssd_mobilenet_v1_coco_2017_11_17"
OD_PATH_TO_CKPT = OD_MODEL_NAME + "/frozen_inference_graph.pb"
OD_PATH_TO_LABELS = os.path.join("tf_models/research/object_detection/data",
                                 "mscoco_label_map.pbtxt")
OD_NUM_CLASSES = 90

od_actor = ObjectDetector.remote(OD_MODEL_NAME, OD_PATH_TO_CKPT,
                                 OD_PATH_TO_LABELS, OD_NUM_CLASSES)

# Initialize objects shared across threads
stream_frame = SharedObject(np.zeros((720, 1280, 3)))

frame = SharedObject(np.zeros((720, 1280, 3)))
lstm_out = SharedObject(np.zeros((360, 640, 3)))
seg_out = SharedObject(np.zeros((360, 640, 3)))
od_out = SharedObject(np.zeros((360, 640, 3)))
frame_renderer = Renderer([frame, lstm_out, seg_out, od_out], stream_frame)
frame_renderer.start()


# Set up web server
video_filename = SharedObject("demo_vids/01.mov")

server = VideoServer(video_filename, stream_frame)
server.start()

decoder = Decoder.remote(video_filename.get())

seg_out_id = None
od_out_id = None
lstm_out_id = None

current_vid_filename = video_filename.get()
while True:
    if video_filename.get() != current_vid_filename:
        current_vid_filename = video_filename.get()
        decoder.set_video.remote(current_vid_filename)

    deadline = timer.remote(1/30)
    frame_id = decoder.next.remote()

    if lstm_out_id is None:
        lstm_out_id = lstm.observe_a_frame.remote(frame_id)
    if seg_out_id is None:
        seg_out_id = seg_actor.segment_image.remote(frame_id)
    if od_out_id is None:
        od_out_id = od_actor.detect_objects.remote(frame_id)

    ready_ids = []
    remaining_ids = [deadline, lstm_out_id, seg_out_id, od_out_id]
    while deadline not in ready_ids:
        ready_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1)

        if lstm_out_id in ready_ids:
            lstm_out.set(ray.get(lstm_out_id))
            lstm_out_id = lstm.observe_a_frame.remote(frame_id)
        if seg_out_id in ready_ids:
            seg_out.set(ray.get(seg_out_id))
            seg_out_id = seg_actor.segment_image.remote(frame_id)
        if od_out_id in ready_ids:
            od_out.set(ray.get(od_out_id))
            od_out_id = od_actor.detect_objects.remote(frame_id)

    frame.set(ray.get(frame_id))
