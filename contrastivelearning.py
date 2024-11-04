import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ImageCoordinatesDataset(Dataset):
    def __init__(self, images, coordinates, transform=None):
        self.images = images
        self.coordinates = coordinates  # list of (latitude, longitude) tuples
        self.transform = transform

    def __getitem__(self, index):
        # Fetch the anchor image and coordinate
        anchor_img = self.images[index]
        anchor_coord = self.coordinates[index]

        # Find a positive sample (close in coordinates)
        pos_index = self._find_positive_sample(index, threshold=0.1)  # example threshold
        pos_img = self.images[pos_index]

        # Find a negative sample (far in coordinates)
        neg_index = self._find_negative_sample(index, threshold=1.0)  # example threshold
        neg_img = self.images[neg_index]

        # Apply transforms if provided
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

    def __len__(self):
        return len(self.images)

    def _find_positive_sample(self, index, threshold):
        # Logic to find positive sample within a distance threshold
        # For simplicity, replace with a basic search, ideally use geospatial distance
        return (index + 1) % len(self.images)

    def _find_negative_sample(self, index, threshold):
        # Logic to find negative sample outside a distance threshold
        return (index + 2) % len(self.images)



class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()  # Remove final classification layer
        self.projection_head = nn.Sequential(
            nn.Linear(512, 128),  # Adjust input size if using a different model
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projection_head(features)
        return projection


def contrastive_loss(anchor, positive, negative, margin=0.5):
    # Calculate cosine similarity
    pos_sim = cosine_similarity(anchor, positive)
    neg_sim = cosine_similarity(anchor, negative)

    # Compute contrastive loss
    loss = torch.clamp(margin - pos_sim + neg_sim, min=0)
    return loss.mean()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load data
dataset = ImageCoordinatesDataset(images=your_images, coordinates=your_coords, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, optimizer
model = ContrastiveModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
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


