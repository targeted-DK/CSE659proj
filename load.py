import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
from geopy.distance import geodesic
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim

model_path = "./contrastive_model.pth"

# Load the trained model weights (if saved) and set the model to evaluation mode
model = ContrastiveModel()
model.load_state_dict(torch.load("contrastive_model.pth"))
model.eval()
model.to(device)

# Load the saved embeddings and coordinates
with open("embeddings_with_coords.pkl", "rb") as f:
    embeddings_with_coords = pickle.load(f)

with open("embeddings_with_coords.pkl", "rb") as f:
    embeddings_with_coords = pickle.load(f)