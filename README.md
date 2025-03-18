# QLDNet-v1
# Ultralytics requirements
# Example: pip install -r requirements.txt

# Base ----------------------------------------
python=3.8.2
matplotlib>=3.3.0
numpy>=1.22.2 # pinned by Snyk to avoid a vulnerability
opencv-python>=4.6.0
pillow>=7.1.2
pyyaml>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.64.0

# Logging -------------------------------------
# tensorboard>=2.13.0
# dvclive>=2.12.0
# clearml
# comet

# Plotting ------------------------------------
pandas>=1.1.4
seaborn>=0.11.0

# Export --------------------------------------
# coremltools>=7.0  # CoreML export
# onnx>=1.12.0  # ONNX export
# onnxsim>=0.4.1  # ONNX simplifier
# nvidia-pyindex  # TensorRT export
# nvidia-tensorrt  # TensorRT export
# scikit-learn==0.19.2  # CoreML quantization
# tensorflow>=2.4.1,<=2.13.1  # TF exports (-cpu, -aarch64, -macos)
# tflite-support
# jax<=0.4.21  # tensorflowjs bug https://github.com/google/jax/issues/18978
# jaxlib<=0.4.21  # tensorflowjs bug https://github.com/google/jax/issues/18978
# tensorflowjs>=3.9.0  # TF.js export
# openvino-dev>=2023.0  # OpenVINO export

# Extras --------------------------------------
psutil  # system utilization
py-cpuinfo  # display CPU info
thop>=0.1.1  # FLOPs computation
# ipython  # interactive notebook
# albumentations>=1.0.3  # training augmentations
# pycocotools>=2.0.6  # COCO mAP
# roboflow

1.download and create environment by conda
2.unzip ultralytics
3.change some path to your own path
4.add datasets as shown in /datasets folds and run train.py(because of the limit of capabilities and privacy of predecessor researcher of the dataset we use,we just put our codes there)
