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



class HDF5ContrastiveDataset(Dataset):
    def __init__(self, hdf5_path, transform=None, pos_threshold=0.5, neg_threshold=5.0):
        # Open the HDF5 file and store paths and coordinates
        self.hdf5_path = hdf5_path
        self.hdf = h5py.File(hdf5_path, 'r')
        self.image_paths = [path.decode('utf-8') for path in self.hdf['image_paths']]
        self.coordinates = list(zip(self.hdf['latitudes'][:], self.hdf['longitudes'][:]))
        self.transform = transform
        self.pos_threshold = pos_threshold  # Distance in km for positive pairs
        self.neg_threshold = neg_threshold  # Distance in km for negative pairs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load the anchor image and coordinate
        anchor_img = self._load_image(self.image_paths[index])
        anchor_coord = self.coordinates[index]

        # Find positive and negative samples
        pos_index = self._find_positive_sample(index)
        neg_index = self._find_negative_sample(index)

        pos_img = self._load_image(self.image_paths[pos_index])
        neg_img = self._load_image(self.image_paths[neg_index])

        # Apply transforms if provided
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

    def _load_image(self, img_path):
        # Load image from path and convert to RGB
        return Image.open(img_path).convert('RGB')

    def _find_positive_sample(self, index):
        anchor_coord = self.coordinates[index]
        for i, coord in enumerate(self.coordinates):
            if i != index and geodesic(anchor_coord, coord).km <= self.pos_threshold:
                return i
        # Fallback: return the next index if no close pair found
        return (index + 1) % len(self.coordinates)

    def _find_negative_sample(self, index):
        anchor_coord = self.coordinates[index]
        for i, coord in enumerate(self.coordinates):
            if i != index and geodesic(anchor_coord, coord).km >= self.neg_threshold:
                return i
        # Fallback: return the next index if no distant pair found
        return (index + 2) % len(self.coordinates)
    
    def close(self):
        # Close the HDF5 file when done
        self.hdf.close()


class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()  # Remove final classification layer
        self.projection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projection_head(features)
        return projection

def contrastive_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    loss = torch.mean(torch.clamp(margin + pos_dist - neg_dist, min=0))
    return loss



# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContrastiveModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create dataset and dataloader
hdf5_path = './image_data.h5'
dataset = HDF5ContrastiveDataset(hdf5_path=hdf5_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for anchor, positive, negative in dataloader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Forward pass
        anchor_out = model(anchor)
        pos_out = model(positive)
        neg_out = model(negative)

        # Compute loss
        loss = contrastive_loss(anchor_out, pos_out, neg_out)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

embeddings_with_coords = []
model.eval()
with torch.no_grad():
    for i in range(len(dataset)):
        img, _, _ = dataset[i]
        img = img.unsqueeze(0).to(device)
        embedding = model(img).cpu().numpy().flatten()
        coord = dataset.coordinates[i]
        embeddings_with_coords.append((embedding, coord))

torch.save(model.state_dict(), "./contrastive_model.pth")

# Save embeddings and coordinates in a pickle file, JSON, or another HDF5 file for future use
# import pickle
# with open("embeddings_with_coords.pkl", "wb") as f:
#     pickle.dump(embeddings_with_coords, f)
