import cv2
import requests
from PIL import Image
from io import BytesIO


def apply_super_resolution(image_path, output_path, model_path):
    # Load the DNN Super-Resolution model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    print(model_path)
    sr.readModel(model_path)
    sr.setModel("edsr", 4)  # Use EDSR model with a scaling factor of 4x
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Upscale the image
    result = sr.upsample(image)
    
    # Save the upscaled image
    cv2.imwrite(output_path, result)

image_path ='/Users/dk/Desktop/Folders/WashU2024/2024 Fall/CSE659A/project/code/project/images/Washington_D.C._0_38.91350357717249_-77.0545545542865.jpg'
output_path = '/Users/dk/Desktop/Folders/WashU2024/2024 Fall/CSE659A/project/code/project/images/Washington_D.C._0_upscaled_38.91350357717249_-77.0545545542865.jpg'
model_path = "./models/EDSR_x4.pb"  # Path to the EDSR model (download from OpenCV repo)

apply_super_resolution(image_path, output_path, model_path)
