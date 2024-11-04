import torch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_paths, texts, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and transform image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tokenize text
        text = self.texts[idx]
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        
        return image, tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize dataset and dataloader (replace with your paths)
dataset = CustomDataset(image_paths, texts, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize models, loss, and optimizer
image_encoder = ImageEncoder().to("cuda")
text_encoder = TextEncoder().to("cuda")
contrastive_loss = ContrastiveLoss()
optimizer = optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)


num_epochs = 10
for epoch in range(num_epochs):
    for images, input_ids, attention_mask in dataloader:
        images, input_ids, attention_mask = images.to("cuda"), input_ids.to("cuda"), attention_mask.to("cuda")

        # Forward pass
        image_embeddings = image_encoder(images)
        text_embeddings = text_encoder(input_ids, attention_mask)
        loss = contrastive_loss(image_embeddings, text_embeddings)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(image_encoder.state_dict(), '/content/drive/MyDrive/CLIP_image_encoder.pth')
torch.save(text_encoder.state_dict(), '/content/drive/MyDrive/CLIP_text_encoder.pth')

