# Install required python packages
# pip3 install -r requirements.txt

# Set up TF models
(cd tf_models/research/ && protoc object_detection/protos/*.proto --python_out=.)

mkdir data

# Download object detection model
(cd data && python3 ../demo_util/download_od_model.py)
