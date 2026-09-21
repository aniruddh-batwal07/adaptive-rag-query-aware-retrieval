import os
import json
import torch
import random
import time
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from src.router.query_classifier import QueryClassifier

class RouterDataset(Dataset):
    def __init__(self, data_path):
        self.queries = []
        self.labels = []
        # 0 = SIMPLE, 1 = COMPLEX
        label_map = {"SIMPLE": 0, "COMPLEX": 1}
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.queries.append(item["query"])
                self.labels.append(label_map[item["label"]])
                
    def __len__(self):
        return len(self.queries)
        
    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="models/router")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("Loading datasets...")
    train_dataset = RouterDataset("datasets/router/train.jsonl")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = RouterDataset("datasets/router/val.jsonl")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    print(f"Initializing model {args.model}...")
    classifier = QueryClassifier(args.model)
    model = classifier.model
    tokenizer = classifier.tokenizer
    device = classifier.device
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    print(f"Training on {device} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (queries, labels) in enumerate(train_loader):
            labels = labels.to(device)
            inputs = tokenizer(queries, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for queries, labels in val_loader:
                labels = labels.to(device)
                inputs = tokenizer(queries, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                outputs = model(**inputs, labels=labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_dataset)
        
        print(f"End of Epoch {epoch+1} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"Saving model to {args.out_dir}...")
    classifier.save(args.out_dir)
    print("Done!")

if __name__ == "__main__":
    train()
