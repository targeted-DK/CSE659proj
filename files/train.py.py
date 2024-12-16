import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import train_test_split
import clip
import json
# from tiny_clip import TinyCLIP
import itertools  
import datetime
import logging

# List of CLIP models 
MODEL_VARIANTS = [
    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    # 'RN101',
    'RN50'
    # "ViT-B/16",
    # "ViT-L/32",
    #  'ViT-B/32', 'ViT-B/16'
    # 'RN50x4'
]

HYPERPARAMETERS = {
    "learning_rate": [1e-4],
    "batch_size": [8],
    "lambda_cls":  
        [0] 
        # [0.25,0.5,
        #            0.75]
}

class ImageTextDataset(Dataset):
    def __init__(self, dataframe, preprocess, tokenizer):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with image paths and text labels.
            preprocess (callable): CLIP image preprocessing function.
            tokenizer (callable): CLIP text tokenization function.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, 'image_path']
        text = self.dataframe.loc[idx, 'prompt']
        class_idx = self.dataframe.loc[idx, 'class_idx']
        
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)
        
        text_tokens = self.tokenizer([text])[0]
        
        return image, text_tokens, class_idx


    
   
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )
    
def train_model(model, hyperparams, train_loader, val_loader, device, geohash_to_idx, idx_to_geohash, NUM_EPOCHS):

    run_dir = f"runs/{model_variant}_lr{hyperparams[0]}_bs{hyperparams[1]}_lambda{hyperparams[2]}"
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(run_dir)
    
    logging.info(f"Starting training: Model={model_variant}, Hyperparams={hyperparams}")
    
 
    num_classes = len(geohash_to_idx)
    print(num_classes)
    class CLIPWithClassification(nn.Module):
        def __init__(self, clip_model, num_classes):
            super(CLIPWithClassification, self).__init__()
            self.clip = clip_model
            self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)
        
        def forward(self, images, prompts):
            image_features = self.clip.encode_image(images)
            text_features = self.clip.encode_text(prompts)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits_per_image = image_features @ text_features.t()  
            logits_per_text = logits_per_image.t()                 
            
         
            class_logits = self.classifier(image_features)       
            return logits_per_image, logits_per_text, class_logits
    
    extended_model = CLIPWithClassification(model, num_classes).to(device)
    
    contrastive_criterion = nn.CrossEntropyLoss()
    classification_criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(extended_model.parameters(), lr=hyperparams[0], weight_decay=0.01)
    
  
    for epoch in range(NUM_EPOCHS):
        extended_model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0
        
        for batch_idx, (images, prompt, geohash_label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")):
            images = images.to(device)
            prompt = prompt.to(device)  # Already tokenized in Dataset
            geohash_label = geohash_label.to(device)
            
            optimizer.zero_grad()
            
            logits_per_image, logits_per_text, class_logits = extended_model(images, prompt)
            
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            
            loss_img = contrastive_criterion(logits_per_image, ground_truth)
            loss_txt = contrastive_criterion(logits_per_text, ground_truth)
            contrastive_loss = (loss_img + loss_txt) / 2
            
            # classification_loss = classification_criterion(class_logits, geohash_label)
            
            # total_loss = contrastive_loss + hyperparams[2] * classification_loss
            total_loss = contrastive_loss
            total_loss.backward()
            optimizer.step()
            
            total_train_loss += total_loss.item()
            
            # Calculate classification accuracy
            _, preds = torch.max(class_logits, 1)
            correct_train += (preds == geohash_label).sum().item()
            total_train_samples += geohash_label.size(0)
        
        # Optionally step the scheduler
        # scheduler.step()
        
      
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train_samples
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        
        # Validation Phase
        extended_model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for images, prompt, geohash_label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                images = images.to(device)
                prompt = prompt.to(device)  
                geohash_label = geohash_label.to(device)
                
              
                logits_per_image, logits_per_text, class_logits = extended_model(images, prompt)
                
              
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                
                # Compute contrastive loss
                loss_img = contrastive_criterion(logits_per_image, ground_truth)
                loss_txt = contrastive_criterion(logits_per_text, ground_truth)
                contrastive_loss = (loss_img + loss_txt) / 2
                
                # Compute classification loss
                # classification_loss = classification_criterion(class_logits, geohash_label)
                
                # Total loss
                # total_loss = contrastive_loss + hyperparams[2] * classification_loss
                total_loss = contrastive_loss
                # Accumulate loss
                total_val_loss += total_loss.item()
                
                # Calculate classification accuracy
                _, preds = torch.max(class_logits, 1)
                correct_val += (preds == geohash_label).sum().item()
                total_val_samples += geohash_label.size(0)
        
        # validation loss and accuracy
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val_samples
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # model checkpoints
        checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}.pth")
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
            # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, 'clip_geohash_contrastive.pth')
    
        logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    logging.info("Training completed successfully.")
    
 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # Load the dataset
    dataframe = pd.read_csv('../image_data2.csv')
    
    required_columns = {'image_path', 'prompt', 'geohash'}
    if not required_columns.issubset(dataframe.columns):
        missing = required_columns - set(dataframe.columns)
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    unique_geohash_labels = dataframe['geohash'].unique().tolist()
    
    geohash_to_idx = {geohash: idx for idx, geohash in enumerate(unique_geohash_labels)}
    idx_to_geohash = {str(idx): geohash for geohash, idx in geohash_to_idx.items()}  
    
    dataframe['class_idx'] = dataframe['geohash'].map(geohash_to_idx)
    
      
    train_df, val_df = train_test_split(dataframe, test_size=0.1, random_state=42, shuffle=True)
    
    for model_variant, hyperparams in itertools.product(MODEL_VARIANTS, itertools.product(*HYPERPARAMETERS.values())):
        # print(model_variant)
        # print(hyperparam_values)
    # hyperparams = {"learning_rate": 1e-4,
    #                     "batch_size": 16,
    #                     "lambda_cls": 0.5
    #     }
        torch.cuda.empty_cache()
        print(model_variant)
        # print(hyperparam_values)
        # model_variant = "ViT-B/32"
        model, preprocess = clip.load(model_variant, device=device)

        train_dataset = ImageTextDataset(train_df, preprocess, clip.tokenize) 
        val_dataset = ImageTextDataset(val_df, preprocess, clip.tokenize)
        
        train_loader = DataLoader(
                    train_dataset,
                    batch_size=hyperparams[1],
                    shuffle=True,
                    num_workers=16 if os.name != 'nt' else 0,
                    drop_last=True
                )
        val_loader = DataLoader(
                val_dataset,
                batch_size=hyperparams[1],
                shuffle=False,
                num_workers=16 if os.name != 'nt' else 0,
                drop_last=False
            )

        num_epochs = 3
        
        train_model(
                model = model,
                hyperparams=hyperparams,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                geohash_to_idx=geohash_to_idx,
                idx_to_geohash=idx_to_geohash,
                NUM_EPOCHS=num_epochs)    


   