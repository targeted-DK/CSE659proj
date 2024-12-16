import time
import torch
from torchvision import transforms
from PIL import Image
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import clip
import torch
import torch.nn as nn
import cv2
import pandas as pd
import geohash2
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import geodatasets

MODEL_PATH = r"C:\Users\mysoo\OneDrive\Documents\GitHub\CSE659proj\files\runs\RN50_lr0.0001_bs16_lambda0.25\epoch_3.pth"  
WATCH_DIRECTORY = r'C:\Users\mysoo\OneDrive\바탕 화면\screenshots'
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")  

model_variant = 'RN50'
learning_rate = 0.0001
batch_size = 16
lambda_val = 0.25

df = pd.read_csv(r"\Users\mysoo\OneDrive\Documents\GitHub\CSE659proj\image_data2.csv")
unique_geohash_labels = df['geohash'].unique().tolist()
geohash_to_idx = {geohash: idx for idx, geohash in enumerate(unique_geohash_labels)}
idx_to_geohash = {idx: geohash for geohash, idx in geohash_to_idx.items()}


class CLIPWithClassification(nn.Module):
        def __init__(self, clip_model, num_classes):
            super(CLIPWithClassification, self).__init__()
            self.clip = clip_model
            self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)
        
        def forward(self, images):
        #     image_features = self.clip.encode_image(images)
        #     text_features = self.clip.encode_text(prompts)
            
        #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        #     logits_per_image = image_features @ text_features.t()  
        #     logits_per_text = logits_per_image.t()                 
            
         
        #     class_logits = self.classifier(image_features)       
        #     return logits_per_image, logits_per_text, class_logits
            # text_tokens = clip.tokenize([""]).to(device)
            image_features = self.clip.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            class_logits = self.classifier(image_features)
            return class_logits

device = "cuda" if torch.cuda.is_available() else "cpu"
     
num_classes = 109

model_clip, preprocess = clip.load(model_variant, device=device)
model = CLIPWithClassification(model_clip, num_classes).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
def run_inference_on_image(image_path):
    print(image_path)
    time.sleep(1)
    img = Image.open(image_path).convert("RGB")
  
    input_image = preprocess(img).unsqueeze(0).to(device)
  
    with torch.no_grad():
        class_logits = model(input_image)


    print(class_logits)
    _, pred = torch.max(class_logits, 1)
    pred_class_idx = pred.item()
    predicted_geohash = idx_to_geohash[pred_class_idx]
    # Directly print the predicted class index since idx_to_geohash is not defined
    print(f"Inference done on {image_path}. Predicted class index: {predicted_geohash}")
    
    lat, lon = geohash2.decode(predicted_geohash)
    print(f"Latitude: {lat}, Longitude: {lon}")
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_global()
    ax.set_extent([-180, 180, -90, 90])
    lat_holder, lon_holder = float(lat), float(lon)
    ax.plot(lon_holder, lat_holder, 'ro', markersize=2, transform=ccrs.PlateCarree())

    plt.title(f"Location for Geohash: {predicted_geohash}")
    plt.show()


    
    # if idx_to_geohash is not None:
    #     pred_label = idx_to_geohash[str(pred_class_idx)]
    #     print(f"Inference done on {image_path}. Predicted class: {pred_label}")
    # else:
    #     print(f"Inference done on {image_path}. Predicted class index: {pred_class_idx}")

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            filename = event.src_path
            # if filename.lower().endswith(IMAGE_EXTENSIONS):
            print(f"New image detected: {filename}")
            run_inference_on_image(filename)



        
def main():
    if not os.path.isdir(WATCH_DIRECTORY):
        raise ValueError(f"The directory {WATCH_DIRECTORY} does not exist.")
    
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    observer.start()
    print(f"Watching directory: {WATCH_DIRECTORY} for new images...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()