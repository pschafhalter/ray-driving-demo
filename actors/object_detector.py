import numpy as np
import ray
import tensorflow as tf

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2


@ray.remote
class ObjectDetector:
    def __init__(self, model_name, path_to_ckpt, path_to_labels, num_classes):
        # Load model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # Load label map
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
                        label_map, max_num_classes=num_classes,
                        use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Initialize tensors
        # with detection_graph.as_default():
        self.sess = tf.Session(graph=self.detection_graph)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name(
                                "image_tensor:0")
        # Each box represents a part of the image where a particular object was
        # detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
                                    "detection_boxes:0")
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
                                    "detection_scores:0")
        self.detection_classes = self.detection_graph.get_tensor_by_name(
                                    "detection_classes:0")
        self.num_detections = self.detection_graph.get_tensor_by_name(
                                    "num_detections:0")

    def detect_objects(self, image):
        image = np.copy(image)
        # Expand dimensions since the model expects images to have shape:
        # [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
          [self.detection_boxes, self.detection_scores, self.detection_classes,
           self.num_detections],
          feed_dict={self.image_tensor: image_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          self.category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        return image
