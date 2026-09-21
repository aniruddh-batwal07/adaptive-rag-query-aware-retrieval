import re
import string
from typing import List, Dict, Any, Union
from collections import Counter
from transformers import AutoTokenizer

_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            # We try to use the SmolLM tokenizer as the standard for token counts across baselines
            _TOKENIZER = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
        except Exception:
            _TOKENIZER = "fallback"
    return _TOKENIZER

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if not isinstance(s, str):
        return ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction: str, reference: str) -> float:
    if not isinstance(prediction, str) or not isinstance(reference, str):
        if prediction == reference:
            return 1.0
        return 0.0
        
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0

def token_f1(prediction: str, reference: str) -> float:
    if not isinstance(prediction, str) or not isinstance(reference, str):
        if prediction == reference:
            return 1.0
        return 0.0

    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
        
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 1.0 if pred_tokens == ref_tokens else 0.0
        
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def count_tokens(text: str) -> int:
    if not text:
        return 0
    
    tokenizer = _get_tokenizer()
    if tokenizer == "fallback":
        return len(text.split())
    else:
        return len(tokenizer.encode(text))

def compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    if original_tokens == 0:
        return 0.0
    return original_tokens / compressed_tokens if compressed_tokens > 0 else float('inf')

def recall_at_k(retrieved_ids: List[str], reference_ids: List[str], k: int) -> float:
    if not reference_ids:
        return float('nan')
    if not retrieved_ids:
        return 0.0
    
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_k).intersection(set(reference_ids))
    return len(relevant_retrieved) / len(reference_ids) if reference_ids else float('nan')

def router_metrics(predictions: List[str], labels: List[str]) -> Dict[str, Any]:
    if not predictions or not labels or len(predictions) != len(labels):
        return {}
    
    tp_c = sum(1 for p, l in zip(predictions, labels) if p == "COMPLEX" and l == "COMPLEX")
    fp_c = sum(1 for p, l in zip(predictions, labels) if p == "COMPLEX" and l == "SIMPLE")
    fn_c = sum(1 for p, l in zip(predictions, labels) if p == "SIMPLE" and l == "COMPLEX")
    tn_c = sum(1 for p, l in zip(predictions, labels) if p == "SIMPLE" and l == "SIMPLE")

    acc = (tp_c + tn_c) / len(predictions) if len(predictions) > 0 else 0.0
    
    prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
    rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
    
    prec_s = tn_c / (tn_c + fn_c) if (tn_c + fn_c) > 0 else 0.0
    rec_s = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
    f1_s = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0.0
    
    macro_f1 = (f1_c + f1_s) / 2
    
    return {
        "accuracy": acc,
        "precision": {"COMPLEX": prec_c, "SIMPLE": prec_s},
        "recall": {"COMPLEX": rec_c, "SIMPLE": rec_s},
        "f1": {"COMPLEX": f1_c, "SIMPLE": f1_s, "macro": macro_f1},
        "confusion_matrix": {
            "TP_COMPLEX": tp_c,
            "FP_COMPLEX": fp_c,
            "FN_COMPLEX": fn_c,
            "TN_COMPLEX": tn_c
        }
    }
