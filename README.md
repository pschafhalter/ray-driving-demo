# Requirements
- PyTorch
- Tensorflow (I recommend the CPU version to avoid CUDA conflicts with PyTorch)
- [Ray](https://github.com/ray-project/ray/)
- protoc

# Setup
1. Clone the repository using `git clone --recursive`.
2. Run `pip3 install -r requirements.txt`.
3. Run `setup.sh`.
4. Download remaining segmentation and control model to `./data`.
    - Copies of models are available [here](https://drive.google.com/open?id=17RsdQD-f_cIUDI_y6JnMPCmgvVL2K_Yo).
5. Add dashcam videos to `./demo_vids`.
    - I recommend videos from the [Berkeley DeepDrive Video Dataset](http://data-bdd.berkeley.edu/),
    but any 1280x720 videos should work.

# Running the application
1. From the project directory, run `python3 demo.py`.
2. The output from the video server should be available at `http://localhost:5000`.

# Example output
I've included some sample output videos [here](https://drive.google.com/drive/folders/1HLfG-55-dP-557UrJAyZG3yVTle-rWTr?usp=sharing).
