from ultralytics import YOLO
import squarify
import matplotlib.pyplot as plt
import cv2
import os
import random
import pandas as pd
import matplotlib.image as mpimg
import seaborn as sns
import torch
sns.set_style('darkgrid')
#%matplotlib inline 
import torch.nn as nn

# Define the paths to the images and labels directories
train_images = "/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/datasets/VTSaR_Crop_640/train/images"
train_labels = "/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/datasets/VTSaR_Crop_640/train/labels"

test_images = "/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/datasets/VTSaR_Crop_640/test/images"
test_labels = "/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/datasets/VTSaR_Crop_640/test/labels"

val_images = "/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/datasets/VTSaR_Crop_640/valid/images"
val_labels = "/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/datasets/VTSaR_Crop_640/valid/labels"

# Get a list of all the image files in the training images directory
image_files = os.listdir(train_images)

# Choose 16 random image files from the list
random_images = random.sample(image_files, 16)

# Set up the plot
fig, axs = plt.subplots(4, 4, figsize=(16, 16))

# Loop over the random images and plot the object detections
for i, image_file in enumerate(random_images):
    row = i // 4
    col = i % 4
    
    # Load the image
    image_path = os.path.join(train_images, image_file)
    image = cv2.imread(image_path)

    # Load the labels for this image
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(train_labels, label_file)
    with open(label_path, "r") as f:
        labels = f.read().strip().split("\n")

    # Loop over the labels and plot the object detections
    # Loop over the labels and plot the object detections
    for label in labels:
        if len(label.split()) != 5:
            continue
        class_id, x_center, y_center, width, height = map(float, label.split())
        x_min = int((x_center - width/2) * image.shape[1])
        y_min = int((y_center - height/2) * image.shape[0])
        x_max = int((x_center + width/2) * image.shape[1])
        y_max = int((y_center + height/2) * image.shape[0])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)


    # Show the image with the object detections
    axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[row, col].axis('off')
# Load an image using OpenCV
image = cv2.imread("/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/datasets/VTSaR_Crop_640/test/images/Video42343.jpg")

# Get the size of the image
height, width, channels = image.shape
print(f"The image has dimensions {width}x{height} and {channels} channels.")
# Loading a pretrained model
model = YOLO("/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/ultralytics/cfg/models/v8/yolov8-SPD-LKA.yaml")
#model = YOLO('/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/runs/detect/train9/weights/last.pt')
#model = YOLO('yolov8n.pt')
#model.load('/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-SPD-Conv/runs/detect/train2-CBAM35/weights/best.pt')
# 检查GPU数量  
'''if torch.cuda.device_count() > 1:  
    print(f"Let's use {torch.cuda.device_count()} GPUs!")  
    # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2 GPUs  
    model = nn.DataParallel(model, device_ids=[0, 1, 2])  
  
# 将模型移动到GPU上  
model.to('cuda:0') '''
#device = torch.device('cuda:2')

# Training the model
model.train(data = '/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/datasets/VTSaR_Crop_640/data.yaml',
            device="7",
            epochs =300,
            single_cls=True,
            cls=0.3,
            imgsz = 640,
            seed = 42,
            batch =1,
            erasing=0.9,
            workers = 4)