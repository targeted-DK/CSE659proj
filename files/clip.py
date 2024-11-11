import clip
import torch
from PIL import Image


model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")


# Create dataset and dataloader
hdf5_path = './image_data.h5'
dataset = HDF5ContrastiveDataset(hdf5_path=hdf5_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Preprocess the image
image = preprocess(Image.open("path_to_image.jpg")).unsqueeze(0).to(device)

# Tokenize the text prompt
text = clip.tokenize(["This is an image of a park", "This is an image of a building"]).to(device)

# Get CLIP's similarity scores
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probabilities:", probs)