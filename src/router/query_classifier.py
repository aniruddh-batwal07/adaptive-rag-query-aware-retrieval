import os
import torch
from typing import List, Dict, Union, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class QueryClassifier:
    """
    Binary classifier to route queries as SIMPLE or COMPLEX.
    """
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self.model_name_or_path = model_name_or_path
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, 
            num_labels=2
        ).to(self.device)
        self.model.eval()
        
        # 0 = SIMPLE, 1 = COMPLEX (alphabetical or standard mapping)
        self.id2label = {0: "SIMPLE", 1: "COMPLEX"}
        self.label2id = {"SIMPLE": 0, "COMPLEX": 1}
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id

    def predict(self, query: str) -> Dict[str, Union[str, float]]:
        """
        Predict complexity for a single query.
        Returns:
            {
                "complexity_label": "SIMPLE" | "COMPLEX",
                "confidence": float
            }
        """
        return self.predict_batch([query])[0]

    def predict_batch(self, queries: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict complexity for a batch of queries.
        """
        if not queries:
            return []
            
        inputs = self.tokenizer(
            queries, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        predictions = []
        for i in range(len(queries)):
            conf, pred_id = torch.max(probs[i], dim=-1)
            label = self.id2label[pred_id.item()]
            predictions.append({
                "complexity_label": label,
                "confidence": conf.item()
            })
            
        return predictions

    def save(self, save_directory: str):
        """
        Save the model and tokenizer to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
