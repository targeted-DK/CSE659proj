import torch
import pickle

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