import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Data Preparation
class ContrastiveDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            img1 = self.transform(image)
            img2 = self.transform(image)
        else:
            img1 = img2 = image

        return img1, img2

# Data transformations
transform_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize dataset and dataloader
image_dir = './images'  # Change this to your images directory
dataset = ContrastiveDataset(image_dir=image_dir, transform=transform_pipeline)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 2: Define the Model
class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ImageEncoder, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection_head(x)
        return x

# Initialize model and optimizer
embedding_dim = 128
model = ImageEncoder(embedding_dim=embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Step 3: Define the NT-Xent Loss (contrastive loss for batch-based learning)
def nt_xent_loss(embeddings, temperature=0.5):
    embeddings = F.normalize(embeddings, dim=1)
    similarity_matrix = embeddings @ embeddings.T
    batch_size = embeddings.shape[0] // 2

    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(embeddings.device)
    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(embeddings.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

    similarity_matrix /= temperature
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

# Step 4: Training Loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for img1, img2 in dataloader:
        img1, img2 = img1.to(device), img2.to(device)

        embeddings1 = model(img1)
        embeddings2 = model(img2)
        
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        loss = nt_xent_loss(embeddings)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Step 5: Visualizing Embeddings with t-SNE
def visualize_embeddings(model, dataloader):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for img1, img2 in dataloader:
            img1 = img1.to(device)
            embeddings = model(img1)
            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, cmap='viridis')
    plt.colorbar()
    plt.title("2D t-SNE projection of embeddings")
    plt.show()

# Step 6: Image Retrieval
def retrieve_similar_images(query_image, model, embeddings, all_images, top_k=5):
    model.eval()
    query_embedding = model(query_image.to(device).unsqueeze(0)).cpu().detach().numpy()
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_k_indices = np.argsort(similarities)[-top_k:]
    similar_images = [all_images[idx] for idx in reversed(top_k_indices)]
    return similar_images

# Example usage of visualization and retrieval
visualize_embeddings(model, dataloader)

# Choose a query image
query_image = dataset[0][0]
all_embeddings = [model(img.to(device).unsqueeze(0)).cpu().detach().numpy() for img, _ in dataset]
similar_images = retrieve_similar_images(query_image, model, np.vstack(all_embeddings), dataset.images)

# Display query and similar images
plt.figure(figsize=(12, 4))
plt.subplot(1, len(similar_images) + 1, 1)
plt.imshow(query_image.permute(1, 2, 0) * 0.5 + 0.5)
plt.title("Query Image")
plt.axis('off')
for i, img in enumerate(similar_images, start=2):
    plt.subplot(1, len(similar_images) + 1, i)
    plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5)
    plt.title(f"Match {i-1}")
    plt.axis('off')
plt.show()
